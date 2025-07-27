import numpy as np
import matplotlib.pyplot as plt

class Oscillator():
    def __init__(self, mass, force_const, initial_amplitude, damping_const=0, driving_force=0, driving_frequency=0):
        self.m = mass
        self.k = force_const
        self.A_0 = initial_amplitude  # Initial position (x0)
        self.b = damping_const
        self.F_0 = driving_force
        self.driving_omega = driving_frequency
        self.eps = 1e-12  # Small epsilon to avoid division by zero

        self.omega_0 = np.sqrt(self.k / self.m)  # Natural frequency
        self.gamma = self.b / (2 * self.m)       # Damping coefficient

        # Determine damping regime
        self.overdamped = lambda: self.gamma > self.omega_0 + self.eps
        self.critically_damped = lambda: np.abs(self.gamma - self.omega_0) < self.eps
        self.underdamped = lambda: self.gamma < self.omega_0 - self.eps

        # For overdamped case: compute beta (≡ omega_d_i)
        if self.overdamped():
            self.beta = np.sqrt(self.gamma**2 - self.omega_0**2)
            # Coefficients for v0 = 0 (no initial velocity)
            denom = 2 * self.beta + self.eps  # Avoid division by zero
            self.C1 = self.A_0 * (self.gamma + self.beta) / denom
            self.C2 = self.A_0 * (self.beta - self.gamma) / denom
        elif self.underdamped():
            self.omega_d = np.sqrt(np.maximum(self.omega_0**2 - self.gamma**2, self.eps))  # Ensure >=0

        # Driving force amplitude/phase (if F_0 > 0)
        if self.F_0 > 0:
            denom = (
                self.m**2 * (self.omega_0**2 - self.driving_omega**2)**2 +
                self.b**2 * self.driving_omega**2 + self.eps
            )
            self.A_driven = self.F_0 / np.sqrt(denom)
            
            # Safe arctan (handles division by zero)
            y = self.b * self.driving_omega
            x = self.m * (self.omega_0**2 - self.driving_omega**2)
            self.delta = np.arctan2(y, x)  # np.arctan2 handles x=0

    def get_x_pos(self, time):
        """Returns position at time(s)."""
        # Homogeneous solution (transient)
        if self.overdamped():
            x_hom = np.exp(-self.gamma * time) * (
                self.C1 * np.exp(self.beta * time) +
                self.C2 * np.exp(-self.beta * time)
            )
        elif self.critically_damped():
            x_hom = (self.A_0 + self.gamma * self.A_0 * time) * np.exp(-self.gamma * time)
        else:  # Underdamped
            x_hom = self.A_0 * np.exp(-self.gamma * time) * np.cos(self.omega_d * time)

        # Driving force (particular solution, only if F_0 > 0)
        x_driven = 0.0
        if self.F_0 > 0:
            x_driven = self.A_driven * np.cos(self.driving_omega * time - self.delta)

        return x_hom + x_driven
    def get_intial_conditions(self):
        return np.array([self.m, self.k, self.A_0, self.b, self.F_0, self.driving_omega])


'''# Test the oscillator

oscillator = Oscillator(
    mass=1,
    force_const=1,
    initial_amplitude=10,
    damping_const=0.2,  # Overdamped (γ > ω0)
    driving_force=0,
    driving_frequency=0
)

t = np.linspace(0, 20, 300)
x = oscillator.get_x_pos(t)

plt.plot(t, x, label="Overdamped (No driving force)")
plt.xlabel("Time")
plt.ylabel("Position")


plt.show()'''

class Harmonic_Oscilator():
        
    def __init__(self, mass, force_const, initial_amplitude, damping_const=0, driving_force=0, driving_frequency=0):
        self.m = mass
        self.k = force_const
        self.A_0 = initial_amplitude  # Initial position (x0)
        self.b = damping_const
        self.F_0 = driving_force
        self.omega_drive = driving_frequency
        #self.p_const = init_phase_const
        self.eps = 1e-12

        self.omega_0 = np.sqrt(self.k / self.m)  # Natural frequency
        self.gamma = self.b / (2 * self.m)       # Damping coefficient

        self.omega_damped =  np.sqrt(self.omega_0 ** 2 - self.gamma ** 2)

        if (self.F_0 != 0):
            self.A = self.F_0/(np.sqrt(self.m**2 * (self.omega_0 ** 2 - self.omega_drive ** 2) + self.b ** 2 * self.omega_drive ** 2) + self.eps)
            self.fp_const = np.arctan2((self.b * self.omega_drive),(self.m * (self.omega_0 ** 2 - self.omega_drive ** 2)))
            print(self.fp_const)
    def get_x_pos(self ,time):
        # Forced
        if (self.F_0 > 0):
            return self.A * (np.cos(self.omega_drive * time - self.fp_const ) - np.exp(-self.gamma * time) * np.cos(self.omega_damped * time - self.fp_const ))
        else:
            return self.A_0 * np.exp(-self.gamma * time) * np.cos(self.omega_damped * time - np.pi/2) # + self.p_const 

class Harmonic_Oscilataor2():
    def __init__(self, mass, force_const, initial_amplitude, damping_const=0, driving_force=0, driving_frequency=0, velocity = 0):
        self.m = mass
        self.k = force_const
        self.A = initial_amplitude  # Initial position (x0)
        self.b = damping_const
        self.F_0 = driving_force
        self.omega_drive = driving_frequency
        self.v = velocity
        #self.p_const = init_phase_const
        #self.eps = 1e-12

        self.omega_0 = np.sqrt(self.k / self.m)  # Natural frequency
        self.gamma = self.b / (2 * self.m)       # Damping coefficient
        self.epsilon = np.sqrt(self.gamma ** 2 - self.omega_0 ** 2)

        #self.omega_damped =  np.sqrt(self.omega_0 ** 2 - self.gamma ** 2)      
        #self.overdamped = lambda: self.gamma > self.omega_0 + self.eps
        self.overdamped = lambda: self.b**2/(4 * self.m ** 2) > self.omega_0 ** 2
        self.crit_damped = lambda: self.omega_0 ** 2 == self.b**2/(4 * self.m ** 2)
        self.light_damped = lambda:  self.b**2/(4 * self.m ** 2) < self.omega_0 ** 2

        if self.overdamped: 
            self.C_1 = (self.A * (self.epsilon - self.gamma) + self.v)/(2 * self.epsilon)
            self.C_2 = (self.A * (self.epsilon + self.gamma) - self.v)/(2 * self.epsilon)
        elif self.crit_damped:
            self.C_1 = self.A
            self.C_2 = self.v - self.gamma * self.A
        elif self.light_damped:
            self.C_1 = self.A
            self.C_2 = (self.v - self.gamma * self.A)/self.epsilon

        self.D_1 = 0.0
        self.D_2 = 0.0

        if (self.F_0 > 0):
            self.D_1 = (self.F_0 * self.omega_drive) / ((self.k - self.omega_drive ** 2 * self.m) ** 2 + (self.omega_drive * self.b) ** 2)
            self.D_2 = (self.F_0 * (self.k - self.omega_drive ** 2 * self.m))/((self.omega_drive * self.b) ** 2 + (self.k - self.omega_drive ** 2 * self.m) ** 2)

    def get_x_pos(self, time):
        self.x_damped = 0.0
        if (self.overdamped):
            self.x_damped = self.C_1 * np.exp((self.gamma + self.epsilon) * time) + self.C_2 * np.exp((self.gamma - self.epsilon) * time)
        elif (self.crit_damped):
            self.x_damped = np.exp(self.gamma * time) * (self.C_1 + self.C_2 * time)
        elif (self.light_damped):
            self.x_damped = np.exp(self.gamma * time) * (self.C_1 * np.cos(self.epsilon * time) + self.C_2 * np.sin(self.epsilon * time))

        self.x_forced = self.D_1 * np.sin(self.omega_drive * time) + self.D_2 * np.cos(self.omega_drive * time)

        self.x_pos = self.x_damped + self.x_forced
        return self.x_pos


oscillator = Harmonic_Oscilataor2(
    mass=1,
    force_const=1,
    initial_amplitude=10,
    damping_const=0.2,  # Overdamped (γ > ω0)
    driving_force=3,
    driving_frequency=1,
    
)

t = np.linspace(0, 20, 300)
x = oscillator.get_x_pos(t)

plt.plot(t, x, label="Overdamped (No driving force)")
plt.xlabel("Time")
plt.ylabel("Position")


plt.show()