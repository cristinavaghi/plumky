import pandas as pd
import os
import numpy as np
from stan_model_definition import StanModelDefinition
from model_definition import Model
import shutil
from backward_prediction import posterior_distribution
import pystan
from stan_model_definition import *
import shelve
from datafiles_series import csv_to_series

import sys
import mlx_py as mlx

from monolix_functions import *

import matplotlib.pyplot as plt
# pandas settings
pd.set_option('display.precision', 3)
pd.set_option('display.float_format', lambda x: '%.3g' % x)
# matplotlib settings
plt.rcParams['text.usetex'] = False
plt.rc('font', family='Arial', size = 18)

def k_fold_monolix(n_sets,
                   validation_folder,
                   models_list,
                   V0,
                   Vc,
                   error_model):
    '''
    Monolix analysis on the n_sets subsets of the initial data set
    Input:
    - n_sets (int): number of subsets
    - validation_folder (string): path of the folder where the subfolders with the subsets are contained
    - models_list (list of strings): list with the names of the models that are used in the monolix analysis
    - V0 (float): value of the initial volume
    - Vc (float): value of Vc (Gompertz model)
    - error_model (string): error model type (e.g. constant, proportional or combined1)
    '''
    for i in range(n_sets):
        k_folder = os.path.join(validation_folder, np.str(i))
        data_path = os.path.join(k_folder,'learning_data.txt')
        control_stack = csv_to_series(data_path)

        monolix_computation(data_path, models_list, os.path.join(k_folder,'monolix_analysis'), V0, Vc, error_model)

        print(np.str((i+1)/n_sets*100) + '%..')
    return

###############################################
def delete_useless_monolix_files(sub_monolix_folder):
    sub = os.listdir(sub_monolix_folder)
    for item in sub:
        if item!='populationParameters.txt':
            if os.path.isfile(os.path.join(sub_monolix_folder, item)):
                os.unlink(os.path.join(sub_monolix_folder, item))
            elif os.path.isdir(os.path.join(sub_monolix_folder, item)):
                shutil.rmtree(os.path.join(sub_monolix_folder, item))

def clean_monolix_workspace(validation_folder, n_sets, models_list):
    '''
    function to delete useless monolix files
    '''
    for i in range(n_sets):
        k_folder = os.path.join(validation_folder, np.str(i))
        for model_name in models_list:
            sub_monolix_folder = os.path.join(k_folder,'monolix_analysis', model_name)
            delete_useless_monolix_files(sub_monolix_folder)

###############################################
def k_fold_prediction(n_sets,
                      validation_folder,
                      models_list,
                      Vc,
                      N,
                      x_label,
                      y_label,
                      tmin):
    '''
    k_fold_prediction:
    function launch the k-fold cross validation (predictions) when the monolix analysis is completed

    Input:
    - n_sets (int): number of sets for the k-fold cross validation
    - validation folder (string): path of the folder where the information are contained: this is needed for the fixed parameter in case of the reduced Gompertz model
    - models_list (list of strings): list with the names of the models that are used
    - Vc (scalar): value of the parameter Vc relative to the Gompertz model
    - N (scalar): int with the number of observations to consider (default 3) --> the last N measurements are taken into account in the Stan analysis
    - x_label (string): name of the x label
    - y_label (string): name of the y label
    - tmin (scalar): minimum time for the time scale in plots
    '''
    backward_folder = os.path.join(validation_folder,'backward_prediction_bi')
    if os.path.isdir(backward_folder) is False:
        os.mkdir(backward_folder)


    for model_name in models_list:
        local_folder = os.path.join(backward_folder, model_name)
        if os.path.isdir(local_folder) is True:
            shutil.rmtree(local_folder)
        os.mkdir(local_folder)

    for model_name in models_list:
        print(model_name)
        model = Model(model_name, Vc = Vc)
        # initialization of the stan models!!!!
        stan_model = StanModelDefinition(model_name,1)
        sm         = pystan.StanModel(model_code = stan_model.model) # precompilation of the stan model
        for i in range(n_sets):
            k_folder = os.path.join(validation_folder, np.str(i))
            test_set = csv_to_series(os.path.join(k_folder,'test_data.txt'))
            monolix_folder = os.path.join(k_folder,'monolix_analysis')
            ########################################
            ######### BACKWARD PREDICTIONS #########
            ########################################

            local_folder = os.path.join(backward_folder, model_name)
            posterior_distribution(test_set,
                                   model,
                                   Vc       = Vc,
                                   N        = N,
                                   folder   = local_folder,
                                   monolix_folder   = monolix_folder,
                                   sm = sm
                                   )
            print(np.str((i+1)/n_sets*100) + '%..')

