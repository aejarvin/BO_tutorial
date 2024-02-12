import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
import math
import scienceplots
from tkinter import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from botorch.models import FixedNoiseGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize

# This is the hidden function. In the context of validating models,
# the python wrapper of the simulation would enter here.
def hidden_f(x):
    y = -1.3*x + 0.8*np.exp(-(0.2 - x)**2/0.01) + 2.0*np.exp(-(0.7 - x)**2/0.003)
    return {'x':x, 'y':y}

# This is a helper function to execute a batch of simulations.
# Since this example is sequential, this would not be needed, but 
# it is retained here to make the code easily applicable to 
# parallel batch execution as well.
def execute_batch(batch):
    cases = []
    for i in range(len(batch)):
        cases.append(hidden_f(batch[i]))
    x_vals = []
    y_vals = []
    for i in range(len(batch)):
        output = cases[i]
        x_vals.append(output['x'])
        y_vals.append(output['y'])
    x_vals = torch.tensor(x_vals)
    y_vals = torch.tensor(y_vals)
    outputdict = {'x':x_vals, 'y':y_vals}
    return outputdict

# This another helper function that updates the dictionary of results.
def append_result_dictionary(result_dictionary, outputdict):
    X_var = np.concatenate([result_dictionary['x'], outputdict['x']])
    Y_var = np.concatenate([result_dictionary['y'], outputdict['y']])
    result_dictionary = {'x':X_var, 'y':Y_var}
    return result_dictionary

# This function updates the GPR surrogate, given the result dictionary.
def surrogate_model_setup(result_dictionary):
    X_var = torch.tensor(result_dictionary['x'])
    Y_var = torch.tensor(result_dictionary['y'])
    outcome_transform = Standardize(m=1)
    input_transform = Normalize(d=1)
    train_Yvar = torch.full_like(Y_var,1e-6)
    gp = FixedNoiseGP(X_var.double().unsqueeze(0).T, Y_var.double().unsqueeze(0).T, 
                      train_Yvar.unsqueeze(0).T,
                      outcome_transform=outcome_transform, 
                      input_transform=input_transform)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    return gp

# This function executes the acquisition function to recommend samples.
def acquire_candidates(gp, bounds, beta=5.0):
    UCB = UpperConfidenceBound(model=gp, beta=beta)
    candidate = optimize_acqf(UCB, bounds=bounds, q=1, num_restarts=10, raw_samples=1024)
    return candidate
    
# This function completes sample collection and updates the figure.
def evaluate_and_plot(fig, canvas):
    global result_dictionary
    if 'plot' not in locals():
        plot = fig.add_subplot(111)
        plot.set_ylim([-2.0,5.0])
        plot.set_xlabel('x')
        plot.set_ylabel('y')
 
    # Update the GPR surrogate
    gp = surrogate_model_setup(result_dictionary)
    # Multiply the exploration parameter to avoid stopping exploration too early.
    # This does not actually work necessarily very well and there are better ways 
    # to do this. A more explorative strategy is for example to put a penalty 
    # on samples collected too close to each other.
    beta = 2*np.log(len(result_dictionary['y']**4)*math.pi**2/0.1)
    candidate = acquire_candidates(gp,bounds,beta=beta)
    if penalty_term:
        if torch.min(torch.abs(candidate[0] - result_dictionary['x'])) < 0.01:
            beta = beta*10
    candidate = acquire_candidates(gp,bounds,beta=beta)
    
    # Plotting
    gp.eval()
    gp.likelihood.eval()
    UCB = UpperConfidenceBound(model=gp,beta=beta)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x_d = torch.linspace(bounds[0,0],bounds[1,0],100)
        x_inp = test_x_d.unsqueeze(1)
        observed_pred = gp.posterior(x_inp)
        lower, upper = observed_pred.confidence_region()
        yt_target = hidden_f(test_x_d)
        plot.plot(yt_target['x'],yt_target['y'],'k--', label='Hidden function')
        fig.savefig('hidden_f.png')
        ucb_pred = UCB(test_x_d.unsqueeze(1).unsqueeze(1))
        plot.plot(test_x_d, ucb_pred, 'r-', label='Acq. function')
        plot.plot(test_x_d.numpy(), observed_pred.mean.numpy(), 'b', label='GPR')
        plot.fill_between(test_x_d.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        plot.plot(result_dictionary['x'], result_dictionary['y'],'ko', label='Samples')
        plot.plot(candidate[0], candidate[1], 'rs')
        plot.legend()
        canvas.draw()
    # Run the case and update the dictionary of results
    outputdict = execute_batch(candidate[0])
    result_dictionary = append_result_dictionary(result_dictionary, outputdict)


if __name__ == '__main__':
    print('Starting the 1D BO tutorial')
    global bounds
    global result_dictionary
    global penalty_term
    # By changing this to True, exploration parameter is increased if
    # samples are collected too close to each other.
    penalty_term = True
    # Setup the boundaries of the search space.
    bounds = torch.stack([
                         torch.tensor([0.0]),
                         torch.tensor([1.0])
             ])
    # Number of initialization points to be selected randomly.
    n_ini = 3
    # Setup a random number generator and collect the initialization set.
    rng = np.random.default_rng()
    ini_set_raw = torch.tensor(rng.random((n_ini, 1)))
    ini_set = bounds[0,:] + ini_set_raw*(bounds[1,:] - bounds[0,:])
    result_dictionary = execute_batch(ini_set)

    # Setup the plotting window.
    window = Tk()
    window.title('Bayesian optimization test')
    fig  = Figure(figsize = (5, 5), dpi = 100)
    plt.style.use(['science','no-latex'])
    canvas = FigureCanvasTkAgg(fig,master = window)
    canvas.get_tk_widget().pack()
    toolbar = NavigationToolbar2Tk(canvas,window)
    toolbar.update()
    canvas.get_tk_widget().pack()
    plot_button = Button(master = window, command = lambda: evaluate_and_plot(fig, canvas), height=2, width=10, text="Sample")
    plot_button.pack()
    window.mainloop()
    

