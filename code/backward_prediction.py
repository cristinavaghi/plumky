import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model_definition import Model
from scipy.optimize import minimize, fmin
import shelve
import shutil
from likelihood_maximization import nonlinear_regression

import seaborn as sns

from stan_model_definition import StanModelDefinition
from monolix_functions import *
import pystan

# ------------------------------
# code to capture pystan output
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])
# ------------------------------

# ______________________________________________________________
# ______________________________________________________________
#
#        BACKWARD PREDICTIONS WITH BAYESIAN INFERENCE
# ______________________________________________________________
# ______________________________________________________________
def define_init_guess(model, n_chains):
    '''
    function to define the initial guess of the parameters for the stan computation.
    The initial guess defined in model_definition are perturbed with a random noise for each chain
    '''
    list_init = [[] for i in range(n_chains)]
    for i in range(n_chains):
        temp = {}
        for j in range(len(model.param_names)):
            temp[('log_'+model.param_names[j])] = np.log(model.initial_guess[j] + model.initial_guess[j]*0.1*np.random.randn(1)[0])
        list_init[i] = temp
    return list_init



# ______________________________________________________________

def posterior_distribution(data,
                            model,
                            Vc,
                            N,
                            folder,
                            monolix_folder,
                            sm,
                            n_chains = 4):
    '''
    posterior_distribution:
    function to compute the posterior distribution of
    the initiation time with Bayesian inference given the last N observation and the
    prior distribution on the parameters

    Input:
    - data (pandas series): set containing the observations of individuals and the time when the measurements are taken
    - model (Model)
    - Vc (scalar): value of the parameter Vc relative to the Gompertz model
    - N (scalar): number of observations to consider (default 3) --> the last N measurements are taken into account in the Stan analysis
    - folder (string): path of the folder where the results are saved
    - monolix_folder (string): path of the folder where the prior information are contained
    - sm (pystan.StanModel): precompiled Stan model
    - n_chains (int): number of chains for the Stan computation

    Results are saved in the folder defined in folder.
    The workspace includes:
    - data: pandas series containing the observations of the individual
    - time: vector of dimension K containing the time steps
    - Y_pred: matrix of dimensions K x n_sample (K is the time length, n_sample is the number of iterations) containing the predicted observations
    - t_init: vector of dimension n_sample with the estimate of the initiation time of the tumor
    - ID: int with the number of the individual relative to the backward prediction
    - N: number of observations considered for the backward predictions (default = 3)

    If an error occurs, a file error.txt is generated with the IDs of the individuals that generated errors
    '''
    for ID in np.unique(data.index.get_level_values(0)):
        try:
            # definition of the data for the stan model
            stan_mod = StanModelDefinition(model_name     = model.model_name,
                                           precompilation = 0,
                                           N              = N,
                                           t              = data[ID].index[-N:],
                                           Y              = data[ID].values[-N:],
                                           t0             = data[ID].index[-N],
                                           V0             = data[ID].values[-N],
                                           monolix_folder = monolix_folder,
                                           Vc             = Vc)

            # initial guess
            init_guess = define_init_guess(model, n_chains)
            with suppress_stdout_stderr():
                fit = sm.sampling(data=stan_mod.data, warmup=1000, iter=2000, chains=n_chains,n_jobs=-1, init=init_guess, refresh = 0)
            Y_pred = fit['Y_pred']
            time   = stan_mod.data['time']
            t_init = fit['t_init']

            # save workspace
            filename = os.path.join(folder, np.str(ID)+'.out')
            my_shelf = shelve.open(filename, 'n') #'n' for new
            my_shelf['Y_pred'] = fit['Y_pred']
            my_shelf['time']   = time
            my_shelf['data']   = data
            my_shelf['ID']     = ID
            my_shelf['N']      = N
            my_shelf['t_init'] = t_init

            # for item in stan_mod.data.keys():
            #     if 'omega_' in item:
            #         my_shelf[item.replace('omega','log')] = fit[item.replace('omega','log')]

            my_shelf.close()

        except RuntimeError:
            f = open(os.path.join(folder,'errors.txt'), 'a+')
            f.write((np.str(ID)+'\n'))
            f.close()

#______________________________________________________________
#______________________________________________________________

# BACKWARD PREDICTIONS WITH LIKELIHOOD MAXIMIZATION
#______________________________________________________________
#______________________________________________________________

def sample_from_std(values, covariance, function, n_sample):
    '''
    sample_from_std:
    function to build prediction interval of the observations when estimates are found with likelihood maximization.
    Input:
    - values: estimates of the parameters found with likelihood maximization
    - covariance: covariance matrix of the estimates found with likelihood maximization
    - function: function depending on the values only (example: function = lambda theta exp(theta*t))
    - n_sample: number of samples
    Output:
    y_pi: matrix containing the sampling of the observations at each time step
    '''
    p = len(values) # number of parameters
    try:
        K = len(function(values)) # time length
    except TypeError:
        K = 1
    try:
        y_pi = np.empty((n_sample, K))
        sample = np.random.multivariate_normal(values, covariance, n_sample)

        for l in range(n_sample):
            y_pi[l,:] = function(sample[l,:])
            if np.any(np.isnan(y_pi[l,:])):
                if np.any(np.logical_not(np.isnan(y_pi[l,:]))):
                    y_pi[l,np.where(np.isnan(y_pi[l,:]))]=0
    except ValueError:
        return np.nan*np.ones((n_sample,K))

    return y_pi

def backward_prediction_likelihood_maximization(data,
                                                validation_folder,
                                                models_list,
                                                Vc,
                                                V0,
                                                N,
                                                x_label,
                                                y_label,
                                                tmin = -20):
    '''
    backward_prediction_likelihood_maximization:
    function to estimate the initiation time with likelihood maximization given the last N observations.
    Individual plots are generated with the backward predictions and the workspace is saved.

    Input:
    - data (pandas series): set containing the observations of individuals and the time when the measurements are taken
    - validation folder (string): path of the folder where the information are contained: this is needed for the fixed parameter in case of the reduced Gompertz model
    - models_list (list of strings): list with the names of the models that are used
    - Vc (scalar): value of the parameter Vc relative to the Gompertz model
    - V0 (scalar): value of the parameter V0 relative to the number of injected cells
    - N (scalar): int with the number of observations to consider (default 3) --> the last N measurements are taken into account in the Stan analysis
    - x_label (string): name of the x label
    - y_label (string): name of the y label
    - tmin (scalar): minimum time for the time scale in plots

    Results are saved in the folder defined in folder.
    The workspace includes:
    - data: pandas series containing the observations of the individual
    - time: vector of dimension K containing the time steps
    - Y_pred: vector of dimension K with the predicted observations (obtained with likelihood maximization)
    - PI_y: matrix of dimensions K x n_sample (K is the time length, n_sample is the number of samples) containing the predicted observations
    - t_init: estimate of the initiation time of the tumor
    - PI_t_init: vector of dimension n_sample with the prediction interval of the initiation time of the tumor
    - ID: int with the number of the individual relative to the backward prediction
    - N: number of observations considered for the backward predictions (default = 3)

    If an error occurs, a file error.txt is generated with the IDs of the individuals that generated errors
    '''
    # plot settings with seaborn
    sns.set()
    sns.set_style("white")
    sns.set_context("paper", font_scale=2.5, rc={"lines.linewidth": 2, 'marker.markersize': 10})
    sns.despine(offset=10, trim=False)
    sns.set_style("ticks")

    # initialization of the folder
    backward_folder = os.path.join(validation_folder,'backward_prediction_lm')
    if os.path.isdir(backward_folder) is False:
        os.mkdir(backward_folder)

    for model_name in models_list:
        # first loop on the models
        local_folder = os.path.join(backward_folder, model_name)
        if os.path.isdir(local_folder) is True:
            shutil.rmtree(local_folder)
        os.mkdir(local_folder)

        for ID in np.unique(data.index.get_level_values(0)):
            # second loop on the individuals in the dataset
            monolix_folder = os.path.join(validation_folder,np.str(ID),'monolix_analysis')
            params = read_pop_parameters(os.path.join(monolix_folder, model_name))
            a_err, b_err, c_err = read_error_model_parameters(os.path.join(monolix_folder, model_name))

            k = 0
            if model_name == 'reduced_gompertz':
                k = params[params['parameter']=='k_pop']['value'].values[0]
            # settings for the nonlinear regression
            Y     = data[ID].values[-N:] # observations
            t     = data[ID].index[-N:]  # time
            # tI and VI are the initial conditions of the function
            tI    = t[0]
            VI    = Y[0]
            sigma1 = a_err
            sigma2 = b_err#*Y # uncertainty on the measurements
            model = Model(model_name, V0=VI, t0=tI, Vc=Vc, k=k) # class Model: it contains the function definition and the parameter settings (bounds, initial guess, fixed parameters,...)

            # nonlinear regression. Results (values, s.e.) are saved in model.params
            res = nonlinear_regression(model, t, Y, sigma1, sigma2)
            if 'Values' in model.params.columns:
                t_init = model.inverse_function(V0, *model.params['Values'].values)
                time = np.linspace(np.min([-20,np.max([-150,t_init])]),t[-1],300)
                Y_pred = model.function(time, *model.params['Values'].values)

                # results are saved in the workspace
                filename = os.path.join(local_folder, np.str(ID)+'.out')
                my_shelf = shelve.open(filename, 'n') #'n' for new
                my_shelf['Y_pred'] = Y_pred
                PI_y = sample_from_std(model.params['Values'].values, model.covariance, lambda p: model.function(time, *p), 4000)
                my_shelf['PI_y']   = PI_y
                my_shelf['time']   = time
                my_shelf['data']   = data
                my_shelf['ID']     = ID
                my_shelf['N']      = N
                my_shelf['t_init'] = t_init
                my_shelf['PI_t_init'] = sample_from_std(model.params['Values'].values, model.covariance, lambda p: model.inverse_function(V0, *p), 4000)
                my_shelf.close()

                # plot of the individual
                p = [[] for i in range(6)]
                fig = plt.figure(dpi = 200)
                ax = fig.add_subplot(1,1,1)
                fig_legend = plt.figure(dpi = 200, figsize=(1.5,0.5))

                y = model.function(time, *model.params['Values'].values)
                y[np.where(np.isnan(y))]=0
                indx = np.argmin(np.abs(time-data[ID].index[-N]))
                p[0], = ax.plot(time[indx:], y[indx:], '-', color = 'royalblue')
                p[1], = ax.plot(time, y,':', label = 'prediction', color = p[0].get_color())
                labels     = ['Fit', 'Prediction']


                y_pi = sample_from_std(model.params['Values'].values, model.covariance, lambda p: model.function(time, *p), 4000)
                while np.any(np.isnan(y_pi)):
                    y_pi = np.delete(y_pi, np.where(np.isnan(y_pi).any(axis = 1))[0][0], axis = 0)
                if len(y_pi) != 0:
                    ax.fill_between(time,
                                     np.percentile(y_pi, 5, axis = 0),
                                     np.percentile(y_pi, 95, axis = 0),
                                     color=p[0].get_color(),
                                     alpha = 0.1)
                p[2],  = ax.fill(np.nan, np.nan, color = p[0].get_color(), alpha = 0.1)
                labels += ['P.I.']
                p[3],  = ax.plot(data[ID].index, data[ID].values, 'ko', mfc = 'none', markersize=10)
                p[4],  = ax.plot(t,Y,'ko', markersize=10)
                labels += ['Data (predictions)']
                labels += ['Data (fit)']


                ax.axvline(0, color = 'k')
                ax.axvline(data[ID].index[-N], color = 'k')
                p[5] = ax.axvline(t_init, color = 'r', linestyle = '--')
                ax.axhline(V0,color = 'k')

                ax.set_yscale('log')
                labels += ['Predicted time']
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)

                xt = ax.get_xticks()

                xtl = xt.tolist()
                xtl=list(["$0$","$t_{n-2}$"])

                xt = [0, data[ID].index[-N]]

                ax.set_xticks(xt)
                ax.set_xticklabels(xtl)

                yt = ax.get_yticks()
                yt = np.append(yt,V0)

                ytl = yt.tolist()
                ytl[-1]="$V_{inj}$"

                ax.set_yticks(yt)
                ax.set_yticklabels(ytl)

                yt = np.array([1e-1,1e1, 1e3])*V0
                yt = np.append(yt,V0)

                ytl = yt.tolist()
                ytl = yt.tolist()
                ytl_str = list()
                for element in ytl:
                    ytl_str.append('%.4g' % element)
                ytl_str[-1]="$V_{inj}$"

                ax.set_yticks(yt)
                ax.set_yticklabels(ytl_str)

                tmax = data[ID].index.max()


                ax.set_xlim((-10,tmax+1))
                textstr = (r'$t_{0,pred}$')
                ax.set_ylim((1e-1*V0,1e5*V0))

                filename = os.path.join(local_folder,np.str(ID)+'.pdf')
                fig.savefig(filename, format = 'pdf', dpi=1000, bbox_inches='tight')
                plt.close('all')
                filelegend = os.path.join(local_folder,'individual_legend.pdf')
            else:
                f = open(os.path.join(local_folder,'errors.txt'), 'a+')
                f.write((np.str(ID)+'\n'))
                f.close()

