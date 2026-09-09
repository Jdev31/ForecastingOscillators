
import math
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from Oscillators import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer

from smac import HyperparameterOptimizationFacade, Scenario


class NeuralNetwork(nn.Module):
    def __init__(self, hidden_neurons=64):
        super().__init__()

        self.linear_tanh_stack = nn.Sequential(

            nn.Linear(1, hidden_neurons),
            nn.Tanh(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.Tanh(),
            nn.Linear(hidden_neurons, 1)
        )

    def forward(self, t):

        logits = self.linear_tanh_stack(t)
        return logits


class CustomLoss(nn.Module):

    def __init__(self, basic_const, physics_const):
        super().__init__()
        self.basic_const = basic_const
        self.physics_const = physics_const

    def forward(self, output, target, residual):
        basic_loss = nn.functional.mse_loss(output, target)
        colloc_loss = nn.functional.mse_loss(residual, torch.zeros_like(residual))
        return self.basic_const * basic_loss + self.physics_const * colloc_loss


# sentinel cost for a diverged/failed trial -- large but finite, since SMAC's
# random-forest surrogate is not guaranteed to handle non-finite inputs
DIVERGED_COST = 1e6


class PINN:
    def __init__(self, curriculum=None, epochs=20000, log_every=5000,
                 use_scheduler=True, n_colloc=2000, split_seed=42):
        m = 1.0
        k = 1.0
        A_0 = 1.0
        v_0 = 0.0
        b = 0.01
        F_0 = 1
        w_0 = 0.8

        oscillator = Harmonic_Oscillator(
            mass=m,
            force_const=k,          # → ω₀ = √(k/m) = 1.0
            initial_amplitude=A_0,    # Start with same amplitude as driven oscillation
            velocity=v_0,             # Zero initial velocity for clean phase
            damping_const=b,       # Very light damping (γ = b/2m = 0.005)
            driving_force=F_0,
            driving_frequency=w_0    # Close to ω₀ = 1.0 → beat frequency ≈ 0.05
        )

        points = 1000
        t_max = 100
        t = np.linspace(0, t_max, points)
        x = oscillator.get_x_pos(t)

        # normalising values around zero so work better in neural network
        t_mean, t_std = float(t.mean()), float(t.std())
        t_norm = (t - t_mean) / t_std

        x_mean, x_std = float(x.mean()), float(x.std())
        x_norm = (x - x_mean) / x_std

        forecast_split = 50.0
        training_values = 100
        batch_size = 100

        np.random.seed(split_seed)
        label_pool = np.where(t <= forecast_split)[0]
        forecast_index = np.where(t > forecast_split)[0]     # returns when elemnet is true in index one, and false in index 2

        # without replacement and over the full pool - randint(0, points - 1) both
        # duplicated indices and could never pick the final point
        training_index = np.random.choice(label_pool, training_values, replace=False)
        interp_index = np.setdiff1d(label_pool, training_index)

        print(f"labels={len(training_index)} drawn from t<={t[label_pool].max():.1f}"
              f"  | interp points={len(interp_index)}  forecast points={len(forecast_index)}")

        def gather(idx):
            return [t_norm[i] for i in idx], [x_norm[i] for i in idx]

        training_data_t, training_data_x = gather(training_index)
        interp_data_t,   interp_data_x   = gather(interp_index)
        forecast_data_t, forecast_data_x = gather(forecast_index)

        def as_tensor(vals):
            return torch.tensor(vals, dtype=torch.float32).reshape(-1, 1)

        # Turned into a tensor to create data to be used in nn
        training_data_t, training_data_x = as_tensor(training_data_t), as_tensor(training_data_x)
        interp_data_t, interp_data_x = as_tensor(interp_data_t), as_tensor(interp_data_x)
        forecast_data_t, forecast_data_x = as_tensor(forecast_data_t), as_tensor(forecast_data_x)

        training_dataset = TensorDataset(training_data_t, training_data_x)
        interp_dataset   = TensorDataset(interp_data_t, interp_data_x)
        forecast_dataset = TensorDataset(forecast_data_t, forecast_data_x)

        training_dataloader = DataLoader(training_dataset, shuffle=True, batch_size=batch_size)

        # full-batch and unshuffled, kept for parity/future diagnostics -- train()'s
        # cost only uses the forecast tensors directly, not this dataloader
        interp_dataloader = DataLoader(interp_dataset, batch_size=max(len(interp_dataset), 1))
        forecast_dataloader = (DataLoader(forecast_dataset, batch_size=len(forecast_dataset))
                               if len(forecast_dataset) else None)

        assert len(forecast_data_t) > 0, "no forecast points -- cost is undefined without a holdout region"

        # physics constants + normalisation stats, needed by ode_residual on every trial
        self.m, self.k, self.b, self.F_0, self.w_0 = m, k, b, F_0, w_0
        self.t_mean, self.t_std = t_mean, t_std
        self.x_mean, self.x_std = x_mean, x_std
        self.t_lo, self.t_hi = float(t_norm.min()), float(t_norm.max())

        # data -- kept on cpu, moved to `device` per-batch in train(), same as the notebook
        self.training_dataloader = training_dataloader
        self.interp_dataloader = interp_dataloader
        self.forecast_dataloader = forecast_dataloader
        self.forecast_data_t, self.forecast_data_x = forecast_data_t, forecast_data_x

        # training knobs, fixed across every SMAC trial
        self.epochs = epochs
        self.log_every = log_every
        self.use_scheduler = use_scheduler
        self.n_colloc = n_colloc

        # curriculum: list of increasing time cutoffs -> training stages. Defaults
        # to a single stage spanning the whole series (no curriculum), which
        # reduces train()'s staged loop to today's plain single-pass behaviour.
        if curriculum is None:
            curriculum = [float(t.max())]
        self.curriculum = curriculum
        self.stages = [(T - self.t_mean) / self.t_std for T in curriculum]
        self.epochs_per_stage = max(1, self.epochs // len(self.stages))

    def ode_residual(self, t_colloc, x_colloc):
        """Dimensionless residual of  m * d2x/dt2 + b * dx/dt + k * x - F_0 cos(w_0 t),
        given the network's normalised output x_colloc = x_norm(t_norm).
        """
        dxdt_norm = torch.autograd.grad(x_colloc, t_colloc,
                                   grad_outputs=torch.ones_like(x_colloc), # means doenst calculate arbritaty calculations between x_i and t_j thoughout nn (saves time)
                                   create_graph=True)[0]

        d2xdt2_norm = torch.autograd.grad(dxdt_norm, t_colloc,
                                   grad_outputs=torch.ones_like(dxdt_norm),
                                   create_graph=True)[0]

        dxdt = dxdt_norm * self.x_std / self.t_std
        d2xdt2 = d2xdt2_norm * self.x_std / self.t_std ** 2

        x_phys = x_colloc * self.x_std + self.x_mean
        t_phys = t_colloc * self.t_std + self.t_mean

        # divided by F_0 so the residual is dimensionless and physics_const is O(1)
        return (self.m * d2xdt2 + self.b * dxdt + self.k * x_phys
                - self.F_0 * torch.cos(self.w_0 * t_phys)) / self.F_0

    def make_colloc(self, n, s_hi=None):
        """Creates tensor on device used, that are random numbers between [0, 1),
        multiplied by the range to give the apropiate number of random colloction points"""
        hi = self.t_hi if s_hi is None else s_hi
        return (torch.rand(n, 1, device=device) * (hi - self.t_lo) + self.t_lo).requires_grad_(True)

    @property
    def configspace(self) -> ConfigurationSpace:

        # Build Configuration Space which defines all parameters and their ranges

        cs = ConfigurationSpace(seed=0)

        # First we create our hyperparameters
        lr = Float("lr", (0.00001, 1), default=0.01, log=True)
        hidden_neurons = Integer("hidden_neurons", (128, 2048), default=512)
        physics_const = Float("physics_const", (0.0, 1), default=0.5)
        basic_const = Float("basic_const", (0.0, 1.0), default=1.0)

        # Add hyperparameters to our configspace
        cs.add([lr, hidden_neurons, physics_const, basic_const])
        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:

        # Set all seeds
        # to make sure is consistent (pytorch is abit relaxed with which generator it uses)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)

        model = NeuralNetwork(config["hidden_neurons"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        loss_fn = CustomLoss(config["basic_const"], config["physics_const"])

        start = time.time()
        try:
            for stage, s_hi in enumerate(self.stages):
                # learning rate needs resetting each stage or CosineAnnealingLR
                # makes the effects of later stages negligible
                for group in optimizer.param_groups:
                    group["lr"] = config["lr"]
                scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.epochs_per_stage) if self.use_scheduler else None)
                if len(self.stages) > 1:
                    print(f"-- stage {stage+1}/{len(self.stages)}: t <= {self.curriculum[stage]:.1f}")

                for epoch in range(self.epochs_per_stage):
                    model.train()
                    for T, X in self.training_dataloader:
                        T, X = T.to(device), X.to(device)

                        # the data term is restricted to the current stage too, so
                        # early stages aren't pulled by labels beyond where physics
                        # is enforced
                        mask = (T <= s_hi).squeeze(-1)

                        t_colloc = self.make_colloc(self.n_colloc, s_hi)
                        x_colloc = model(t_colloc)
                        residual = self.ode_residual(t_colloc, x_colloc)   # always -- no use_physics gate

                        if mask.any():
                            loss = loss_fn(model(T[mask]), X[mask], residual)
                        else:

                            loss = config["physics_const"] * nn.functional.mse_loss(
                                residual, torch.zeros_like(residual))

                        if not torch.isfinite(loss):
                            print(f"  [DIVERGED] stage {stage+1} epoch {epoch+1}: loss={loss.item()}")
                            return DIVERGED_COST

                        # Backpropagation
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                    if scheduler is not None:
                        scheduler.step()
                    if epoch % self.log_every == 0 or epoch == self.epochs_per_stage - 1:
                        print(f"  epoch {epoch+1:>6}/{self.epochs_per_stage}  loss {loss.item():.6f}"
                              f"  lr {optimizer.param_groups[0]['lr']:.2e}")
                        
        except Exception as exc:
            print(f"  [FAILED] hidden={config['hidden_neurons']} lr={config['lr']:.2e} -> {exc!r}")
            return DIVERGED_COST

        # cost = forecast-region data MSE, matching the convention already used by
        # the "Running Lots of data" sweep cell in the notebook
        model.eval()
        with torch.no_grad():
            cost = nn.functional.mse_loss(
                model(self.forecast_data_t.to(device)), self.forecast_data_x.to(device)).item()
        if not math.isfinite(cost):
            return DIVERGED_COST

        print(f"  hidden={config['hidden_neurons']:>4} lr={config['lr']:.2e} "
              f"physics_const={config['physics_const']:.3f} basic_const={config['basic_const']:.3f} "
              f"-> forecast MSE {cost:.6f}  ({time.time()-start:.1f}s)")
        return cost


if __name__ == "__main__":
    pinn = PINN()

    # Next, we create an object, holding general information about the run
    scenario = Scenario(
        pinn.configspace,
        n_trials=100,  # We want to run max 50 trials (combination of config and seed)
        deterministic=True,  # every (config, seed) here is fully reproducible (fixed
       )

    # We want to run the facade's default initial design, but we want to change the number
    # of initial configs to 5.
    initial_design = HyperparameterOptimizationFacade.get_initial_design(scenario, n_configs=5)

    # Now we use SMAC to find the best hyperparameters
    smac = HyperparameterOptimizationFacade(
        scenario,
        pinn.train,
        initial_design=initial_design,
        overwrite=True,  # If the run exists, we overwrite it; alternatively, we can continue from last state
    )

    incumbent = smac.optimize()

    # Get cost of default configuration
    default_cost = smac.validate(pinn.configspace.get_default_configuration())
    print(f"Default cost: {default_cost}")

    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost: {incumbent_cost}")
