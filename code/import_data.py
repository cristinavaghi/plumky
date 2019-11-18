import pandas as pd
import numpy as np
import os
import sys
from datafiles_series import *


def divide_learning_test_groups(data_set, N = 2):
    '''''
    data_set: set with all the individuals
    N: number of subset of the data set
    '''''
    indices = np.unique(data_set.index.get_level_values(0))

    #np.random.shuffle(indices)

    sets_list = [[] for i in range(N)]

    num_ind_set = np.int(np.ceil(len(indices)/N))

    for i in range(N):
        groups_idx = indices[num_ind_set*i:min(num_ind_set*(i+1),len(indices))]
        set_loc = pd.DataFrame()
        set_loc = set_loc.stack()

        for idx in groups_idx:
            loc = data_set[data_set.index.get_level_values(0) == idx]
            set_loc = set_loc.append(loc)
        sets_list[i] = set_loc
    return sets_list


def create_k_subsets(control_stack,
                     n_sets,
                     validation_folder,
                     data_name,
                     N=3):
    sets = divide_learning_test_groups(control_stack, n_sets)

    for i in range(n_sets):
        k_folder = os.path.join(validation_folder, np.str(i))
        if os.path.isdir(k_folder) is False:
            os.mkdir(k_folder)
        test_set = sets[i]
        learning_set = pd.DataFrame()
        learning_set = learning_set.stack()
        for j in range(len(sets)):
            if i!=j:
                learning_set = learning_set.append(sets[j])
        learning_data_path = os.path.join(k_folder,'learning_data.txt')
        test_data_path = os.path.join(k_folder,'test_data.txt')
        if os.path.isfile(learning_data_path) is False:
            series_to_csv(learning_data_path, learning_set, N)
        if os.path.isfile(test_data_path) is False:
            series_to_csv(test_data_path, test_set, N)



def import_data(data_name, folder,  N=3):
    '''
    function to import data.
    In the directory foldername/data_name the subfolders are created:
    - global_monolix_analysis: subfolder containing "data.txt" - file containing the entire dataset (format used by Monolix).
    - n subfolders are created (with n = number of individuals) to perform k-fold cross validation. Each subfolder has the name of the individual that is
      used in the test set. In each subfolder there are is the "learning_data.txt" with the observations that are used to learn the population parameters with monolix
      and the "test_data.txt" with the observations of the individual considered in the test set to perform backward predictions.

    Input:
    - data_name (string): string with the name of the data set ("MDA-MB-231_volume", "MDA-MB-231_fluorescence" or "lung_volume")
    - folder (string): name of the directory where the files are saved
    - N (int): number of observations to consider in the backward predictions

    Output:
    - control_stack (pandas series): data set containing the observations for each individual
    - V0 (scalar): value relative to the number of injected cells converted in the appropriate unit
    - Vc (scalar): value of the parameter Vc relative to the Gompertz model
    - x_label (string): name of the x label
    - y_label (string): name of the y label
    - n_sets (int): number of subsets that are created for the k-fold cross validation (default: number of sets = number of individuals in the dataset)
    - lambda_alpha (scalar): value of the in vitro proliferation rate relative to the

    '''
    validation_folder = os.path.join(folder, data_name)
    if data_name == 'MDA-MB-231_fluorescence':
        x_label = 'Time (days)'
        y_label = 'Fluorescence (phot./s)'
        ratio = 1.519e+07 # [(phot./s)/m^3] ratio between fluorescence and volume data
        V0 = ratio*0.08
        Vc = ratio*0.08
        lambda_alpha = 0.965

        filepath = '../data/breast_fluo_data.txt'
        control_stack = csv_to_series(filepath)
        n_sets = len(np.unique(control_stack.index.get_level_values(0)))
        if os.path.isdir(validation_folder) is False:
            os.mkdir(validation_folder)
            create_k_subsets(control_stack, n_sets, validation_folder, 0, data_name)

    elif data_name == 'MDA-MB-231_volume':
        x_label = 'Time (days)'
        y_label = 'Volume (mm$^3$)'
        V0 = 1
        Vc = 1
        lambda_alpha = 0.837

        filepath = '../data/breast_vol_data.txt'
        control_stack = csv_to_series(filepath)
        n_sets = len(np.unique(control_stack.index.get_level_values(0)))
        if os.path.isdir(validation_folder) is False:
            os.mkdir(validation_folder)
            create_k_subsets(control_stack, n_sets, validation_folder, 0, data_name)
    elif data_name == 'lung_volume':
        x_label = 'Time (days)'
        y_label = 'Volume (mm$^3$)'
        V0 = 1
        Vc = 1
        lambda_alpha = 0.929

        filepath = '../data/lung_vol_data.txt'
        control_stack = csv_to_series(filepath)
        n_sets = len(np.unique(control_stack.index.get_level_values(0)))
        if os.path.isdir(validation_folder) is False:
            os.mkdir(validation_folder)
            create_k_subsets(control_stack, n_sets, validation_folder, 0, data_name)
    else:
        raise ValueError('Wrong input')
    control_stack = control_stack.drop(control_stack[control_stack.values==0].index)

    global_analysis_folder = os.path.join(validation_folder, 'global_monolix_analysis')
    if os.path.isdir(global_analysis_folder) is False:
        os.mkdir(global_analysis_folder)
    series_to_csv(os.path.join(global_analysis_folder,'data.txt'), control_stack)

    #################################

    return control_stack, V0, Vc, x_label, y_label, n_sets, lambda_alpha
