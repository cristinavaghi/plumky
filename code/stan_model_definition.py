import numpy as np
import os
import pandas as pd
from monolix_functions import *

def gompertz():
    model = """
            data {
              int<lower=0> N;
              real t[N];
              real Y[N];
              real Vc;
              // the following data define the prior
              real t0;
              real V0;
              real t_min; // minimum time to make backward predictions
              vector[2] params_mean;
              matrix[2,2] params_covariance;

              // data relative to the error model
              real a_err;
              real b_err;
              real c_err;

              int n_vec; // array dimension
              real time[n_vec];
            }
            parameters {
              //individual parameters that have to be estimated
              vector[2] params;
              real VI;
            }
            transformed parameters {
              real sigma[N];
              real alpha0 = exp(params[1]);
              real beta = exp(params[2]);
              real m[N];
              for (i in 1:N) {
                m[i] = Vc*exp(alpha0/beta)*exp(exp(-beta*(t[i]-t0))*(log(VI/Vc)-alpha0/beta));
                sigma[i] = a_err+b_err*pow(m[i],c_err);
              }
            }
            model {
              // priors
              params ~ multi_normal(log(params_mean), params_covariance);
              VI ~ normal(V0, a_err+b_err*V0^c_err);
              // likelihood
              Y ~ normal(m, sigma);
            }

            generated quantities{
              real Y_mean[n_vec];
              real Y_pred[n_vec];
              real s[n_vec];
              real t_init;
              t_init = t0 - 1/beta*log((-alpha0/beta)/(log(VI/Vc)-alpha0/beta));
              for(i in 1:n_vec){
                // Posterior parameter distribution of the mean
                Y_mean[i] = Vc*exp(alpha0/beta)*exp(exp(-beta*(time[i]-t0))*(log(VI/Vc)-alpha0/beta));
                // Posterior predictive distribution
                s[i] = a_err+b_err*pow(Y_mean[i],c_err);
                Y_pred[i] = Y_mean[i];
            }
            }

        """
    return model

def reduced_gompertz():
    model = """
            data {
              int<lower=0> N;
              real t[N];
              real Y[N];
              real Vc;

              real t0;
              real V0;
              real t_min; // minimum time to make backward predictions

              // parameters relative to the linear regression
              real k;
              real q;

              // the following data define the prior
              real beta_pop;
              real omega_beta;

              // data relative to the error model
              real a_err;
              real b_err;
              real c_err;

              int n_vec; // array dimension
              real time[n_vec];
            }
            parameters {
              //individual parameters that have to be estimated
              real log_beta;
              real VI;
            }
            transformed parameters {
              real sigma[N];
              real beta = exp(log_beta);
              real alpha0 = q + k*beta;
              real m[N];
              for (i in 1:N) {
                m[i] = Vc*exp(alpha0/beta)*exp(exp(-beta*(t[i]-t0))*(log(VI/Vc)-alpha0/beta));
                sigma[i] = a_err+b_err*pow(m[i],c_err);
              }
            }
            model {
              // priors
              log_beta ~ normal(log(beta_pop), omega_beta);
              VI ~ normal(V0, a_err+b_err*V0^c_err);
              // likelihood
              Y ~ normal(m, sigma);
            }
            generated quantities{

              real Y_mean[n_vec];
              real Y_pred[n_vec];
              real s;
              real t_init;
              t_init = t0 - 1/beta*log((-alpha0/beta)/(log(VI/Vc)-alpha0/beta));
              for(i in 1:n_vec){
                // Posterior parameter distribution of the mean
                Y_mean[i] = Vc*exp(alpha0/beta)*exp(exp(-beta*(time[i]-t0))*(log(V0/Vc)-alpha0/beta));
                // Posterior predictive distribution
                s = a_err+b_err*pow(Y_mean[i],c_err);
                Y_pred[i] = Y_mean[i];
            }
            }
        """
    return model

class StanModelDefinition:
    '''
    StanModelDefinition is a class to set the stan model.
    Objects:
        - model (string): model name
        - data (dict): it contains all the data relative to an individual to perform backward predictions
    '''
    def __init__(self,
                 model_name,
                 precompilation,
                 N              = [],
                 t              = [],
                 Y              = [],
                 t0             = [],
                 V0             = [],
                 monolix_folder = [],
                 Vc             = [],
                 tmin           = -20,
                 n_vec          = 300):
        '''
        Constructor
        Input:
            - model_name (string): name of the model
            - precompilation (bool)
            - N (int): number of observations (it must be equal to the cardinality of Y and of t)
            - t (numpy vector): time when the measurements are taken
            - Y (numpy vector): vector of observations
            - t0, V0 (real, real): initial condition
            - tmin (real)
            - monolix_folder (string): global monolix folder containing the folder model_name with the results of the monolix analysis
            - Vc (real): value of Vc for the Gompertz models
            - n_vec: length of the time array for the predictions
        '''
        if precompilation == 0:
            params = read_pop_parameters(os.path.join(monolix_folder, model_name))
            a_err, b_err, c_err = read_error_model_parameters(os.path.join(monolix_folder, model_name))
            self.model_name = model_name
            time = np.linspace(tmin, t[-1], n_vec)

            # a_err = 0 # only proportional error model is considered!

            if model_name =='gompertz':
                mean = np.array([params[params['parameter']=='alpha0_pop']['value'].values[0],
                                 params[params['parameter']=='beta_pop']['value'].values[0]])

                var_alpha = params[params['parameter']=='omega_alpha0']['value'].values[0]
                var_beta = params[params['parameter']=='omega_beta']['value'].values[0]
                corr_alpha_beta = params[params['parameter']=='corr_beta_alpha0']['value'].values[0]

                covariance = np.array([[var_alpha,corr_alpha_beta*np.sqrt(var_alpha*var_beta)],
                              [corr_alpha_beta*np.sqrt(var_alpha*var_beta),var_beta]])
                self.model = gompertz()
                self.data = {'N': N,
                          't': t,
                          'Y': Y,
                          'Vc': Vc,
                          't0':  t0,
                          'V0': V0,
                          't_min': tmin,
                          'params_mean': mean,
                          'params_covariance': covariance,
                          'a_err':    a_err,
                          'b_err':    b_err,
                          'c_err':    c_err,
                          'n_vec': n_vec,
                          'time': time
                          }

            elif model_name =='reduced_gompertz':
                self.model = reduced_gompertz()
                self.data = {'N': N,
                          't': t,
                          'Y': Y,
                          'Vc': Vc,
                          'k': params[params['parameter']=='k_pop']['value'].values[0],
                          'q': params[params['parameter']=='q_pop']['value'].values[0],
                          't0':  t0,
                          'V0': V0,
                          't_min': tmin,
                          'beta_pop':   params[params['parameter']=='beta_pop']['value'].values[0],
                          'omega_beta':  params[params['parameter']=='omega_beta']['value'].values[0],
                          'a_err':    a_err,
                          'b_err':    b_err,
                          'c_err':    c_err,
                          'n_vec': n_vec,
                          'time': time
                          }
        else:
            if model_name =='gompertz':
                self.model = gompertz()
            elif model_name == 'reduced_gompertz':
                self.model = reduced_gompertz()








