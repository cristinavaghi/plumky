import os
import shutil
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# pandas settings
pd.set_option('display.precision', 3)
pd.set_option('display.float_format', lambda x: '%.3g' % x)
# matplotlib settings
plt.rcParams['text.usetex'] = False
plt.rc('font', family='Arial', size = 18)
plt.rcParams['text.usetex'] = False
from import_data import import_data
from monolix_functions import monolix_computation
from k_fold_cross_validation import k_fold_monolix, clean_monolix_workspace, k_fold_prediction, monolix_computation
from backward_prediction import backward_prediction_likelihood_maximization
from create_prediction_summary import *


tmin = -20 # to set the time scale for backward predictions
##################################################
##################################################
##               INITIALIZATION                 ##
##################################################
##################################################

# directory creation and data importation
if os.path.isdir(folder_results) is False:
    os.mkdir(folder_results)
control_stack, V0, Vc, x_label, y_label, n_sets, lambda_alpha = import_data(cell_line_name, folder_results)
validation_folder = os.path.join(folder_results, cell_line_name)

##################################################
##################################################
##              MONOLIX ANALYSIS                ##
##################################################
##################################################
if run_global_monolix_analysis == 1:
    print("Global Monolix analysis")
    global_monolix_folder = os.path.join(validation_folder,'global_monolix_analysis')
    monolix_computation(os.path.join(global_monolix_folder,'data.txt'), models_list, global_monolix_folder, V0, Vc, error_model, x_label, y_label, lambda_alpha)
    if os.path.isfile(os.path.join(global_monolix_folder,'summary.pdf')):
        shutil.copy(os.path.join(global_monolix_folder,'summary.pdf'), os.path.join(validation_folder,'summary_global_population_analysis.pdf'))
    if os.path.isfile(os.path.join(global_monolix_folder,'correlation.pdf')):
        shutil.copy(os.path.join(global_monolix_folder,'correlation.pdf'), os.path.join(validation_folder,'gompertz_correlation.pdf'))
        shutil.copy(os.path.join(global_monolix_folder,'correlation_box.pdf'), os.path.join(validation_folder,'gompertz_correlation_box.pdf'))

##################################################
##################################################
##               k-fold MONOLIX                 ##
##################################################
##################################################

# k-fold cross validation: in this part the population analysis on the learning set is performed
if run_k_fold_monolix == 1:
    print("k-fold Monolix analysis")
    k_fold_monolix(n_sets, validation_folder, models_list, V0, Vc, error_model)

# function to clean the monolix workspace in order to delete useless files (only population parameters remain)
if clean_monolix_workspace == 1:
    clean_monolix_workspace(validation_folder, n_sets, models_list)

##################################################
##################################################
##                 PREDICTIONS                  ##
##################################################
##################################################
# k-fold predictions with bayesian inference
if run_k_fold_predictions_bi == 1:
    print("Backward predictions with Bayesian inference")
    k_fold_prediction(n_sets, validation_folder, models_list, Vc, N, x_label, y_label, tmin)
    backward_folder = os.path.join(validation_folder,'backward_prediction_bi')
    plot_bp_bayes(control_stack, models_list, backward_folder, V0, x_label, y_label)

# predictions with likelihood maximization
if run_predictions_lm == 1:
    print("Backward predictions with likelihood maximization")
    backward_prediction_likelihood_maximization(control_stack, validation_folder, models_list, Vc, V0, N, x_label, y_label, tmin)

# graphics of backward predictions
if create_graphics == 1:
    figures_folder = os.path.join(validation_folder,'prediction_summary')
    if os.path.isdir(figures_folder) is False:
        os.mkdir(figures_folder)

    backward_folder = os.path.join(validation_folder,'backward_prediction_bi')
    err_t0_bi, PI_t0_bi, t0_in_PI_bi = read_results(models_list, backward_folder, n_sets, control_stack, 'bi', V0)

    limit_outlier = 100
    for c in err_t0_bi.columns:
        mask = np.abs(getattr(err_t0_bi,c)) > limit_outlier
        err_t0_bi.loc[mask, c] = np.nan

    swarm_plot(err_t0_bi, 'bi', models_list, figures_folder)

    backward_folder = os.path.join(validation_folder,'backward_prediction_lm')
    err_t0_lm, PI_t0_lm, t0_in_PI_lm = read_results(models_list, backward_folder, n_sets, control_stack, 'lm', V0)

    for c in err_t0_lm.columns:
        mask = np.abs(getattr(err_t0_lm,c)) > limit_outlier
        err_t0_lm.loc[mask, c] = np.nan

    swarm_plot(err_t0_lm, 'lm', models_list, figures_folder)
    save_results(figures_folder, err_t0_bi, err_t0_lm,PI_t0_bi, PI_t0_lm)

    if os.path.isfile(os.path.join(figures_folder,'summary_backward_predictions.pdf')):
        shutil.copy(os.path.join(figures_folder,'summary_backward_predictions.pdf'), os.path.join(validation_folder,'summary_backward_predictions.pdf'))

# function to clean the backward predictions workspace in order to delete useless files (only plots remain)
if clean_backward_workspace == 1:
    for model_name in models_list:
        b_folder = os.path.join(validation_folder, 'backward_prediction_bi', model_name)
        sub = os.listdir(b_folder)
        for item in sub:
            if '.out.db' in item:
                if os.path.isfile(os.path.join(b_folder, item)):
                    os.unlink(os.path.join(b_folder, item))

        b_folder = os.path.join(validation_folder, 'backward_prediction_lm', model_name)
        sub = os.listdir(b_folder)
        for item in sub:
            if '.out.db' in item:
                if os.path.isfile(os.path.join(b_folder, item)):
                    os.unlink(os.path.join(b_folder, item))
