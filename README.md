# Creating a Physics Informed Neural Network that forecasts the movements of a Forced Damped Harmonic Oscillator

This repository primarily focuses on the jupyter notebook `pytorch_PINN.ipynb` which trains a Physics Informed Neural Network (PINN) to forecast the simple, but crucial physical model of a Harmonic Oscillator. The model is trained on a limited number of data points taken from analytic harmonic oscillator solver script (`Oscillators.py`) and an additional number of collocation data points that are used to inform the neural network of the physical dynamics described by the harmonic oscillator equation: 

$m\frac{{d}^2x}{{dt^2}} + b \frac{dx}{dt} + kx - F_0 \cos({w_0 t}) = 0$

The network is trained on the equation above by first calculating  $\frac{\mathrm{d}^2x}{\mathrm{dt^2}}$ and $\frac{dx}{dt}$ by backpropgating from the predicted x value of the model to the initial `t` value using chain rule. Then using these results you can find the mean squared error of the equation and treat it as an additional loss value. 


## Results 
<!-- 
![Example Comparison](OscillatorTimeSeries/figures/forcast_comparison1.png)

The figure shows the difference in forecasting between the basic Neural Network (blue) and the PINN (orange) compared to the analytical solution (grey) -->

I found that adding the physics informed loss function drastically improved its forecasting ability. It was also noted that with enough training time, the model could fit the data to the correct analytical solution without any data points, being solely trained on the harmonic oscillator equation, however it was a much longer training time. These results were expected as basic neural networks are known for not being able to forecast data as they have no effective memory and the PINN intuitively increases its forecasting ability by forcing the model to learn the physical properties behind it which repeats with time. 

## The purpose of this repository

This project was my first time training a machine learning model, I decided on this specific project because I wanted learn how a neural network works and a project implementing a basic neural network would consolidate my learning. Additionally I had heard about PINN's and realized it would fit in well with an analytical model of a harmonic oscillator I had created a couple of years earlier. With both the basic neural network being able to fit to the harmonic oscillator I thought of two options to see the PINNs additional use: adding noise to the data or forecasting the data into the future. I chose forecasting with the plan to add noise on top of this. The main idea behind using a PINN was it is a method of supplementing a small dataset with more data that could be generated using the physical model the data should follow.  


## Challenges faced and future improvements

As I had limited knowledge of what would be the best hyperparameters to use for the model, I wanted a method to automate finding a good combination of hyperparameters. Additionally I did not want find it via a grid search method as the 5-6 dimensional space would get large very quickly and likely a lot time would be wasted on training obviously bad models.
I came across Baysian optimisation, where a surrogate model would be used to estimate the optimal hyperparameter's in the hyperparameter space, that would be updated with every new point trained.

To optimise the hyperparameters, I implemented SMAC3's random forest baysian optimisation algorithm. Where the surrogate model would be a random forest that was trained on differnet hyperparameter combinations, the predicted average and variance for the forest for potential future hyperparameter configurations were then calculated to work out what is the optimal next configuration to try.

The code (found in `Config_variable_training.py`) has been written to vary: learning rate, number of neurons in the hidden layer of the model, and physical models importance compared to the actual data in training. However, I have not spent enough time running and testing the algorithm to get confident results before I have had to move back to university where I do not have access to my desktop and GPU to train the models, leading to an unfortunate end to this project for now.

In future I plan to continue to test the hyperparameter optimisation algorithm to check reproducibility and accuracy; increase the dimensions of the hyperparemeter space; add noise to the model to see if this model would be useful when looking at real world data; and test multiple different harmonic oscillators to see if the optimal hyperparameters transfer between them. As with my other projects I plan to make a longer version of this describing the PINN in more detail when I can collect slightly more data, which can be found on my website at https://jdev31.github.io

## Credits
Before beginning this project I started my neural network journey watching 3Blue1Brown's  Youtube videos on Backpropagation. I then moved onto Andrew Karpathy's "Neural Networks: Zero to Hero" Lecture on Youtube, implementing a neural network from scratch and modeling it off of the pytorch architecture. From here I read the first PINN paper: Raissi, Perdikaris & Karniadakis (2019), "Physics-Informed Neural Networks." and then implemented my own PINN using the Introduction to Pytorch Lecture series and Karpathy's lecture as a basis to build off.