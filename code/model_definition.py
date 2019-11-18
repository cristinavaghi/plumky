import numpy as np
import pandas as pd
from numpy import exp, log, sqrt, isinf

# exponential
def exponential_function(time, t0, V0, alpha):
    time = time-t0
    return V0*np.exp(time*alpha)

#logistic
def logistic_function(time, t0, V0, lambda0, lambda1):
    time = time-t0
    return V0*lambda1/((lambda1-V0)*exp(-lambda0*time)+V0)

# gompertz
def gompertz_function(time, t0, V0, alpha0, beta, Vc):
    time = time-t0
    V = Vc*(V0/Vc)**(np.exp(-beta*time))*np.exp(alpha0/beta*(1-np.exp(-beta*time)))
    return V

def gompertz_inverse(V,t0,V0,alpha0,beta,Vc):
    s1 = np.log(V/Vc)-alpha0/beta
    s2 = np.log(V0/Vc)-alpha0/beta
    return t0 - 1/beta*np.log(s1/s2)

# reduced gompertz
def reduced_gompertz_function(time, t0, V0, beta, q, k, Vc):
    time = time-t0
    alpha0 = k*beta+q
    V = Vc*(V0/Vc)**(np.exp(-beta*time))*np.exp(alpha0/beta*(1-np.exp(-beta*time)))
    return V

def reduced_gompertz_inverse(V, t0, V0, beta, q, k, Vc):
    alpha0 = k*beta + q
    s1 = np.log(V/Vc)-alpha0/beta
    s2 = np.log(V0/Vc)-alpha0/beta
    return t0 - 1/beta*np.log(s1/s2)



class Model:
    '''
    The class Model contains the information relative to the different models
    that we consider.
    Initial parameters are given.
    Methods for the estmation of the parameters are set (MLE/fixed parameters)
    and fixed effects parameters
    '''

    def __init__(self,
                 model_name,
                 V0 = 1,
                 t0 = 0,
                 Vc = 1,
                 q = 0,
                 k = 0):
        '''
        Constructor.
        Input:
        - model_name: name of the model that is considered
        - V0: initial volume
        - Vc: parameter for the Gompertz model
        - q, k: parameters for the reduced Gompertz model s.t. alpha = k*beta + q
        '''
        lower_bounds = []
        upper_bounds = []
        fixed_values = []
        fixed_effects = []
        inverse_function = []
        if model_name == 'exponential':
            plot_name = 'Exponential'
            function = lambda time, alpha: exponential_function(time, t0, V0, alpha)
            param_names = ['a']
            initial_guess = [0.3]
            initial_guess_bounds = [(0,1)]
            bounds = list([(0., np.inf),(0., np.inf)])
            all_initial_values = initial_guess.copy()
            fixed_effects = fixed_values.copy()

        elif model_name == 'logistic':
            plot_name = 'Logistic'
            function = lambda time, lambda0, lambda1: logistic_function(time, t0, V0, lambda0, lambda1)
            param_names = ['a', 'K']
            initial_guess = [0.2, 2e3*Vc]#[V0, 0.09, 1e10*Vc]
            initial_guess_bounds = [(0,1), (0,1e25*Vc)]
            bounds = list([(0., np.inf),(0., np.inf)])
            all_initial_values = initial_guess.copy()
            fixed_effects = fixed_values.copy()

        elif model_name == 'gompertz':
            plot_name = 'Gompertz'
            function = (lambda time, alpha0, beta: gompertz_function(time, t0, V0, alpha0, beta, Vc))
            inverse_function = (lambda V, alpha0, beta: gompertz_inverse(V, t0, V0, alpha0, beta, Vc))
            param_names = ['alpha0', 'beta']
            initial_guess = [0.6, 0.07]
            initial_guess_bounds = [(0.3,0.9), (0.03,0.13)]
            bounds = list([(0., np.inf),(0., np.inf)])
            fixed_values  = ['Vc']
            all_initial_values = initial_guess.copy()
            all_initial_values.extend([Vc])
            fixed_effects = fixed_values.copy()

        elif model_name == 'reduced_gompertz':
            plot_name = 'Reduced Gompertz'
            function = (lambda time, beta: reduced_gompertz_function(time, t0, V0, beta, q, k, Vc))
            inverse_function = (lambda V, beta: reduced_gompertz_inverse(V, t0, V0, beta, q, k, Vc))
            if k == 0:
                param_names = ['beta','k']
                initial_guess = [0.07, 7]
                initial_guess_bounds = [(0.03,0.13),(0,10)]
                bounds = list([(0., np.inf),(0., np.inf)])
                fixed_values  = ['Vc', 'q']
                all_initial_values = initial_guess.copy()
                all_initial_values.extend([Vc, q])
                fixed_effects = fixed_values.copy()
                fixed_effects.append('k')
            else:
                param_names = ['beta']
                initial_guess = [0.07]
                initial_guess_bounds = [(0.03,0.13)]
                bounds = list([(0., np.inf)])
                fixed_values  = ['Vc', 'q', 'k']
                fixed_effects = fixed_values.copy()
                all_initial_values = initial_guess.copy()
                all_initial_values.extend([Vc, q, k])


        else:
            raise ValueError('Unknown model')
        all_initial_values.extend([V0, t0]) # to add initial value of the volume and initial time
        fixed_values.extend(['V0','tin'])
        fixed_effects.extend(['V0','tin'])

        self.model_name           = model_name # model name (string without spaces)
        self.plot_name            = plot_name # plot title
        self.function             = function # lambda function of the model
        self.inverse_function     = inverse_function # lambda function of the inverse model (when backward prediction is performed)
        self.param_names          = param_names # list with the names of the parameters of the model that have to be estimated
        self.initial_guess        = initial_guess # initial guess for the Monolix analysis
        self.initial_guess_bounds = initial_guess_bounds # bounds of initial guesses when they are sampled
        self.covariance           = [] # covariance of the estimates for likelihood maximization
        self.bounds               = bounds # bounds of the parameters (not used)
        self.fixed_values         = fixed_values # list of parameters that are not estimated
        self.fixed_effects        = fixed_effects # list of parameters with fixed effects
        self.all_initial_values   = all_initial_values # list pf the values of all the parameters (also the ones that are not estimated)
        self.error                = 1
        self.warn                 = 0
        self.params = pd.DataFrame(self.param_names, columns = ['Parameters']) # table with the estimates of likelihood maximization

