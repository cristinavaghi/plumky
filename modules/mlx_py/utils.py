#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
# MODULES
# import packages for interface between R and Python
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from collections import OrderedDict

# import matplotlib.pyplot and set plot linewidth and font
import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 2
plt.rc('font', family='Arial', size = 20)
plt.rcParams['text.usetex'] = False

# import pandas and set display precision
import pandas as pd
pd.set_option('display.precision', 3)
pd.set_option('display.float_format', lambda x: '%.3g' % x)
import numpy as np
import os
import sys
import shutil
import pdb
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
def check_txt_file(filename):
    if os.path.isfile(filename) is False or ('.txt' not in filename and '.csv' not in filename and '.dat' not in filename):
        raise ValueError((filename + ' is not a valid file for Monolix.'))

def set_model(model):
    try:
        check_txt_file(model)
    except ValueError:
        dir = os.path.dirname(sys.modules[__name__].__file__)
        if '.txt' not in model:
            model = model + '.txt'
        model_file = os.path.join(dir,'models',model)
        if os.path.isfile(model_file) is False or '.txt' not in model_file:
            print(model_file)
            raise
        model = model_file
    return model
'''
convertion of Python variables into R variables:
Input:   python_variable
Output:  r_conversion_of_python_variable
'''
#----
# parameter names
def set_parameter_names_2r(parameter_names):
    for i in range(len(parameter_names)):
        if '_pop' not in parameter_names[i] and 'omega_' not in parameter_names[i] and 'beta_' not in parameter_names[i]:
            parameter_names[i] = parameter_names[i] + '_pop'
    return robjects.StrVector(parameter_names)
# covariates
def set_covariates_2r(covariates):
    if len(covariates)==0 or type(covariates) is not dict:
        return robjects.vectors.ListVector({})
    od = OrderedDict()
    for key in covariates.keys():
        od[key] = robjects.vectors.ListVector(covariates[key])
    covariates2r = robjects.vectors.ListVector(od)
    return covariates2r
#----
# header types
def set_header_types_2r(header_types):
    if len(header_types)!=0:
        if type(header_types) is not list:
            raise ValueError('header_types must be of type list')
    return robjects.StrVector(header_types)
#----
# scenario tasks
def set_scenario_tasks_2r(scenario_tasks):
    if len(scenario_tasks)!=0:
        if type(scenario_tasks) is not dict:
            raise ValueError('scenario_tasks must be of type dict')
    return robjects.ListVector(scenario_tasks)
#----
# list of plot names
def set_plot_list_2r(plot_list):
    if len(plot_list)!=0:
        if type(plot_list) is not list:
            raise ValueError('plot_list must be of type list')
    return robjects.StrVector(plot_list)
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
# LAUNCH MONOLIX
#-------------------------------------------------------------------------------------------------------------
def launch_monolix(
    data_path,
    model,
    project_path,
    parameter_names       = [],
    initial_values        = [],
    fixed_params          = [],
    params_fixed_effects  = [],
    covariates            = {},
    correlated_parameters = [],
    error_model           = [],
    scenario_tasks        = [],
    observation_type      = "continuous",
    header_types          = list(["ID", "TIME","OBSERVATION"]),
    plot_list             = [],
    linearization         = False ):
    '''
    function that converts the variables in input in R variables and launches the R API version monolix (Monolix_R_API.R)
    Monolix_R_API.R must be located in the package folder mlx_py
    Input:
    - data_path: path of the file containing the data file (format: *.txt. The header in the file must be: [ID Time Observation])
    - model: path of the file containing the model (format: *.txt) or string with the name of a specific model declared in the folder "models"
        (i.e. one of the following: compartmental1, compartmental2, exponential_linear, exponential, gompertz, logistic, power_law)
    - project_path: path of the file of the project (format: *.mlxtran)
    - parameter_names: list of the name of the variables (default empty)
    - initial_values: list with the initial values of the parameters  (default empty)
    - fixed_parameters: list with the names of the fixed parameters  (default empty)
    - params_fixed_effects: list with the names of the parameters that have only fixed effects  (default empty)
    - covariates: dict of dict specifying the covariates.
      Example: covariates = {"V0": {"VI" : True}, "t0": {"tI": True}}
    - correlated_parameters: list of correlated parameters
    - error_model: string with the type of error model (e.g. "constant" or "combined"; default empty)
    - scenario_tasks: dictionary that sets the output of monolix  (default empty)
    - observation_type: string with the type of observation (default empty)
    - header_types: list with the names of the headers (default empty)
    - plot_list: list of plot that which data are saved  (default empty)
    - linearization: binary variable that sets if the linearization in Monolix computation has to be done (default False)
    NOTE that the default values then defined in Monolix_R_API.R. Check the line where run_monolix(...) is called to see if any variable has to be added as input
    '''
    # conversion of the input variables in R variables
    check_txt_file(data_path)
    model_path             = set_model(model)
    parameter_names2r      = set_parameter_names_2r(parameter_names)
    initial_values2r       = robjects.FloatVector(initial_values)
    fixed_params2r         = set_parameter_names_2r(fixed_params)
    params_fixed_effects2r = robjects.StrVector(params_fixed_effects)
    covariates2r           = set_covariates_2r(covariates)
    correlated_p2r         = robjects.StrVector(correlated_parameters)
    header_types2r         = set_header_types_2r(header_types)
    scenario_tasks2r       = set_scenario_tasks_2r(scenario_tasks)
    plot_list2r            = set_plot_list_2r(plot_list)

    # definition of the command "source" from R
    source            = robjects.r['source']

    # find path of the directory of the project
    r_path = os.path.dirname(sys.modules[__name__].__file__)
    # upload script Monolix_R_API.R
    source(os.path.join(r_path,'Monolix_R_API.R'))
    # definition of the command "run_monolix" from R (run_monolix is a function defined in Monolix_R_API.R)
    run_monolix       = robjects.r['run_monolix']
    # call run monolix
    run_monolix(
        data_path,
        model_path,
        project_path,
        parameter_names2r,
        initial_values2r,
        fixed_params2r,
        params_fixed_effects2r,
        covariates2r,
        correlated_p2r,
        error_model,
        header_types,
        observation_type) # NOTE that you can add more terms here. Check the file Monolix_R_API.R
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
# GET MONOLIX RESULTS
#-------------------------------------------------------------------------------------------------------------
def get_monolix_results(folder_path):
    '''
    the function returns the results of the monolix computation.
    Input: folder_path: path of the folder containing the results (such as 'populationParameters.txt')
    Output:
    - pop_param: estimated parameters (median values, standard errors f the fixed and the random effects and of the parameters defining the error model)
    '''
    pop_param_file = os.path.join(folder_path,'populationParameters.txt')
    pop_param      = pd.read_csv(pop_param_file, sep = ',')

    loglike_file   = os.path.join(folder_path,'LogLikelihood/logLikelihood.txt')
    log_like       = pd.read_csv(loglike_file, sep = ',')

    return pop_param, log_like
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
# PLOTS
#-------------------------------------------------------------------------------------------------------------
# INDIVIDUAL PLOTS
#-------------------------------------------------------------------------------------------------------------
def plot_individual_fits(
    folder_input,
    folder_output,
    x_label,
    y_label,
    plot_prediction_interval=True,
    use_ID_for_fig_name=True
    ):
    if (os.path.isdir(folder_input) is False ) or (os.path.isdir(folder_output) is False ):
        raise ErrorValue('The paths relative to the folders are not a valid directories')

    graphic_data_path = os.path.join(folder_input,'ChartsData','IndividualFits')
    file_list = os.listdir(graphic_data_path)
    for item in file_list:
        if '_fits.txt' in item:
            file_fits    = os.path.join(graphic_data_path,item)
            obs_name     = item.split('_fits.txt')[0]
    df_fits      = pd.read_csv(file_fits, sep = ',')
    file_obs     = os.path.join(graphic_data_path, obs_name + '_observations.txt')
    df_obs       = pd.read_csv(file_obs, sep  = ',')
    ID_vec       = df_fits['ID'].unique()
    for comp, ID in enumerate(ID_vec):
        time       = df_fits[df_fits['ID'] == ID].loc[: ,'time'].values
        fit        = df_fits[df_fits['ID'] == ID].loc[: ,'indivPredMean'].values
        time_obs   = df_obs[df_obs['ID']   == ID].loc[: ,'time'].values
        obs        = df_obs[df_obs['ID']   == ID].loc[: , obs_name].values
        y_low      = df_obs[df_obs['ID']   == ID].loc[: ,'piLower'].values
        y_up       = df_obs[df_obs['ID']   == ID].loc[: ,'piUpper'].values
        fig        = plt.figure(dpi = 200)
        ax         = fig.add_subplot(111)
        lines      = [[] for i in range(3)]
        fig_legend = plt.figure(dpi = 200, figsize=(1,0.5))
        lines[0],  = ax.plot(time, fit, color = 'royalblue')
        lines[1],  = ax.plot(time_obs, obs, 'k.')
        labels     = ['Individual fit', 'Observed data']
        if plot_prediction_interval:
            ax.fill_between(time_obs, y_low, y_up, color = 'royalblue', alpha = 0.1)
            lines[2],  = ax.fill(np.nan, np.nan, color = 'royalblue', alpha = 0.1)
            labels    += 'Prediction interval'
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if use_ID_for_fig_name:
            file_output = os.path.join(folder_output, (np.str(ID) + '.pdf'))
        else:
            file_output = os.path.join(folder_output, (np.str(comp+1) + '.pdf'))
        fig.savefig(file_output, format='pdf', dpi=1000, bbox_inches='tight')
        plt.close(fig)
        if ID==ID_vec[0]:
            fig_legend.legend(handles   = lines,
                              labels    = ['Individual fit', 'Observed data'],
                              loc       = 'center',
                              edgecolor = 'none')
            file_legend = os.path.join(folder_output, ('legend.pdf'))
            fig_legend.savefig(file_legend, format='pdf', dpi=1000, bbox_inches='tight')
            plt.close(fig_legend)
#-------------------------------------------------------------------------------------------------------------
# VPC
#-------------------------------------------------------------------------------------------------------------
def plot_vpc(
    folder_input,
    folder_output,
    x_label,
    y_label):
    '''
    Plot VPC
    '''
    if (os.path.isdir(folder_input) is False ) or (os.path.isdir(folder_output) is False ):
        raise ErrorValue('The paths relative to the folders are not a valid directories')
    graphic_data_path = os.path.join(folder_input,'ChartsData','VisualPredictiveCheck')
    file_list         = os.listdir(graphic_data_path)
    for item in file_list:
        if '_observations.txt' in item:
            file_obs     = os.path.join(graphic_data_path,item)
            obs_name     = item.split('_observations.txt')[0]
    df_obs     = pd.read_csv(file_obs, sep = ',')
    file_vpc   = os.path.join(graphic_data_path, obs_name+'_percentiles.txt')
    df_vpc     = pd.read_csv(file_vpc, sep = ',')
    fig        = plt.figure(dpi = 200)
    ax         = fig.add_subplot(111)
    lines      = [[] for i in range(6)]
    fig_legend = plt.figure(dpi = 200, figsize = (1,0.5))
    bins       = df_vpc['bins_middles'].values
    # predicted medians
    theoretical_median = df_vpc['theoretical_median_median'].values
    theoretical_lower  = df_vpc['theoretical_lower_median'].values
    theoretical_upper  = df_vpc['theoretical_upper_median'].values
    lines[0],          = ax.plot(bins, theoretical_median, '-', color='salmon', linewidth = 2)
    lines[1],          = ax.plot(bins, theoretical_lower,'-', color='royalblue', linewidth = 2)
    ax.plot(bins, theoretical_upper, '-', color='royalblue', linewidth = 2)

    file_new_bins = os.path.join(graphic_data_path,obs_name+'_bins.txt')

    # empirical percentiles
    empirical_median = df_vpc['empirical_median'].values
    empirical_lower  = df_vpc['empirical_lower'].values
    empirical_upper  = df_vpc['empirical_upper'].values
    lines[2], = ax.plot(bins, empirical_median, '--', color='black', linewidth = 2)
    ax.plot(bins, empirical_lower,'--', color='black', linewidth = 2)
    ax.plot(bins, empirical_upper, '--', color='black', linewidth = 2)

    # prediction intervals
    df_new_bins = pd.read_csv(file_new_bins, sep=',')
    bins = np.concatenate([[df_new_bins['bins_values'].values[0]], bins, [df_new_bins['bins_values'].values[-1]]])
    # median
    low_v = df_vpc['theoretical_median_piLower'].values
    up_v  = df_vpc['theoretical_median_piUpper'].values
    low_v = np.concatenate([[low_v[0]],low_v,[low_v[-1]]])
    up_v = np.concatenate([[up_v[0]], up_v, [up_v[-1]]])
    ax.fill_between(bins, low_v, up_v, color = 'salmon', alpha = 0.1)
    lines[3], = ax.fill(np.nan,np.nan, color = 'salmon', alpha = 0.1)

    # 90
    low_v = df_vpc['theoretical_upper_piLower'].values
    up_v  = df_vpc['theoretical_upper_piUpper'].values
    low_v = np.concatenate([[low_v[0]],low_v,[low_v[-1]]])
    up_v = np.concatenate([[up_v[0]], up_v, [up_v[-1]]])
    ax.fill_between(bins, low_v, up_v, color = 'royalblue', alpha = 0.1)
    lines[4], = ax.fill(np.nan,np.nan, color = 'royalblue', alpha = 0.1)
    lines[5], = ax.plot(df_obs['time'].values, df_obs[obs_name].values, 'k.', markersize = 8)#, markersize=2, alpha = 0.5)
    #y_lim = ax.get_ylim()
    #ax.set_xlim(x_lim)
    #ax.set_ylim(y_lim)
    #10
    low_v = df_vpc['theoretical_lower_piLower'].values
    up_v  = df_vpc['theoretical_lower_piUpper'].values
    low_v = np.concatenate([[low_v[0]],low_v,[low_v[-1]]])
    up_v = np.concatenate([[up_v[0]], up_v, [up_v[-1]]])
    ax.fill_between(bins, low_v, up_v, color = 'royalblue', alpha = 0.1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    file_output = os.path.join(folder_output,'vpc.pdf')
    fig.savefig(file_output, format='pdf', dpi=1000, bbox_inches='tight')

    fig_legend.legend(handles   = lines,
                      labels    = ('Predicted median 50%',
                                   'Predicted median 10% and 90%',
                                   'Empirical percentiles',
                                   'P.I. 50%',
                                   'P.I. 10% and 90%',
                                   'Data'),
                      loc       = 'center',
                      edgecolor = 'none')
    file_legend = os.path.join(folder_output,'legend.pdf')
    fig_legend.savefig(file_legend, format='pdf', dpi=1000, bbox_inches='tight',transparent=True)
    plt.close('all')
#------------------------------
#  PLOT OBSERVATIONS VS PREDICTIONS
def plot_obs_vs_pred( folder_input, folder_output):
    if (os.path.isdir(folder_input) is False ) or (os.path.isdir(folder_output) is False ):
        raise ErrorValue('The paths relative to the folders are not a valid directories')
    graphic_data_path = os.path.join(folder_input, 'ChartsData', 'ObservationsVsPredictions')
    file_list         = os.listdir(graphic_data_path)
    for item in file_list:
        if '_obsVsPred.txt' in item:
            file_obs     = os.path.join(graphic_data_path, item)
            obs_name     = item.split('_obsVsPred.txt')[0]
    df_obs            = pd.read_csv(file_obs, sep = ',')
    file_res          = os.path.join(graphic_data_path, (obs_name+'_visualGuides.txt'))
    df_res            = pd.read_csv(file_res, sep = ',')
    if len(df_obs[obs_name].values) > 100:
        plt.rcParams['lines.markersize'] = 5
    else:
        plt.rcParams['lines.markersize'] = 10
    fig        = plt.figure(dpi = 200)
    ax         = fig.add_subplot(111)
    lines      = [[] for i in range(3)]
    fig_legend = plt.figure(dpi = 200, figsize = (1, 0.5))

    lines[0], = ax.plot(df_obs['indivPredMode'].values, df_obs[obs_name].values,
                        '.', color='royalblue')

    lines[1], = ax.plot(df_res['indivPred_ci_abscissa'].values,
                        df_res['indivPred_ci_abscissa'].values,
                        '-', color='black')

    lines[2], = ax.plot(df_res['indivPred_ci_abscissa'].values,
                    df_res['indivPred_piLower'].values,
                    '--', color='black')
    ax.plot(df_res['indivPred_ci_abscissa'].values,
                        df_res['indivPred_piUpper'].values,
                        '--', color='black')
    ax.set_xlabel('Individual predictions')
    ax.set_ylabel('Observations')
    file_output = os.path.join(folder_output,'obs_vs_pred.pdf')
    fig.savefig(file_output, format='pdf', dpi=1000, bbox_inches='tight')

    fig_legend.legend(handles   = lines,
                      labels    = ('Observed data',
                                   'y = x',
                                   '90% prediction interval'),
                      loc       = 'center',
                      edgecolor = 'none')
    file_legend = os.path.join(folder_output,'legend.pdf')
    fig_legend.savefig(file_legend, format='pdf', dpi=1000, bbox_inches='tight')
    plt.close('all')
    plt.rcParams['lines.markersize'] = 6 # reset default value

def plot_prediction_distribution(
    folder_input,
    folder_output,
    x_label,
    y_label):
    '''
    Plot prediction distribution
    '''
    if (os.path.isdir(folder_input) is False ) or (os.path.isdir(folder_output) is False ):
        raise ErrorValue('The paths relative to the folders are not a valid directories')
    graphic_data_path = os.path.join(folder_input,'ChartsData','PredictionDistribution')
    file_list         = os.listdir(graphic_data_path)
    for item in file_list:
        if '_observations.txt' in item:
            # pdb.set_trace()
            file_obs     = os.path.join(graphic_data_path, item)
            obs_name     = item.split('_observations.txt')[0]

    df_obs     = pd.read_csv(file_obs, sep = ',')
    file_perc  = os.path.join(graphic_data_path, obs_name+'_percentiles.txt')
    df_prd     = pd.read_csv(file_perc, sep = ',')
    fig = plt.figure(dpi = 200)
    ax         = fig.add_subplot(111)
    lines      = [[] for i in range(3)]
    fig_legend = plt.figure(dpi = 200, figsize = (1,0.5))

    lines[0], = ax.plot(df_obs['time'].values, df_obs[obs_name].values, 'k.', markersize = 8)
    lines[1], = ax.plot(df_prd['time'].values, df_prd['median'].values, 'k')

    vector_perc = [5,15,25,35,45]

    for i in range(0,len(vector_perc)):
        ax.fill_between(df_prd['time'].values,
                                    df_prd['p'+np.str(vector_perc[i])].values,
                                    df_prd['p'+np.str(100-vector_perc[i])].values,
                                    color = 'green',
                                    alpha = 0.1*(i+0.5) )
    lines[2], = ax.fill(np.nan,np.nan, color = 'green',alpha = 0.1*(len(vector_perc)+0.5) )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    file_output = os.path.join(folder_output,'prediction_distribution.pdf')
    fig.savefig(file_output, format='pdf', dpi=1000, bbox_inches='tight')

    fig_legend.legend(handles   = lines,
                      labels    = ('Observed data',
                                   'Median',
                                   'P. I.'),
                      loc       = 'center',
                      edgecolor = 'none',
                      fontsize = 12)
    file_legend = os.path.join(folder_output,'legend.pdf')
    fig_legend.savefig(file_legend, format='pdf', dpi=1000, bbox_inches='tight')
    plt.close('all')

def plot_residuals(
    folder_input,
    folder_output,
    xlabel,
    ylabel,
    ylim = None,
    ):

    if (os.path.isdir(folder_input) is False ) or (os.path.isdir(folder_output) is False ):
        raise ErrorValue('The paths relative to the folders are not a valid directories')
    graphic_data_path = os.path.join(folder_input,'ChartsData','ScatterPlotOfTheResiduals')
    file_list         = os.listdir(graphic_data_path)
    for item in file_list:
        if '_residuals.txt' in item:
            # pdb.set_trace()
            filepath     = os.path.join(graphic_data_path, item)
            obs_name     = item.split('_residuals.txt')[0]

    df = pd.read_csv(filepath)
    time = df['time'].values
    V = df['prediction_iwRes_mode'].values
    iwres = df['iwRes_mode'].values
    fig = plt.figure(dpi = 200)
    ax = fig.add_subplot(1,1,1)
    ax.plot(time, iwres,'.', color = 'royalblue', markersize = 10)
    ax.plot([time.min(), time.max()],np.zeros(2), 'k--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('IWRES')
    if ylim == None:
        g_ylim = ax.get_ylim()
        m_ylim = np.max(np.abs(g_ylim))
        ylim   = (-m_ylim,m_ylim)
    ax.set_ylim(ylim)
    filename = os.path.join(folder_output,'iwres_time.pdf')
    fig.savefig(filename, dpi=1000, format = 'pdf', bbox_inches='tight')

    fig = plt.figure(dpi = 200)
    ax = fig.add_subplot(1,1,1)
    ax.plot(V, iwres,'.', color = 'royalblue', markersize = 10)
    ax.plot([V.min(), V.max()],np.zeros(2), 'k--')
    ax.set_xlabel(ylabel)
    ax.set_ylabel('IWRES')
    ax.set_ylim(ylim)
    filename = os.path.join(folder_output,'iwres_obs.pdf')

    fig.savefig(filename, dpi=1000, format = 'pdf', bbox_inches='tight')
#------------------------------
#  SAVE PLOTS
def save_monolix_graphics(
        project_folder,
        x_label,
        y_label,
        plot_list=[],
        plot_PI_indiv=True,
        use_ID_for_fig_name=True
        ):
    '''
    the function generates the plots starting from the data defined in the project_folder/ChartsData and saves them in project_folder/Graphics
    Input:
    - project_folder: path of the folder containing the results of the monolix computation (the folder must contain the subfolder "ChartsData")
    - x_label: string defining the x label
    - y_label: string defining the y label
    - plot_list: list of plots that are created (default all)
    '''
    graphic_folder = os.path.join( project_folder, 'Graphics' )
    if os.path.isdir(graphic_folder) is True:
        shutil.rmtree(graphic_folder)
    os.mkdir(graphic_folder)

    if plot_list==[]:
        plot_list = ['IndividualFits', 'VisualPredictiveCheck', 'ObservationsVsPredictions', 'PredictionDistribution', 'ScatterPlotOfTheResiduals']
    for item in plot_list:
        if item == 'IndividualFits':
            folder_output = os.path.join(graphic_folder, item)
            if os.path.isdir(folder_output) is False:
                os.mkdir(folder_output)
            plot_individual_fits(
                project_folder,
                folder_output,
                x_label,
                y_label,
                plot_prediction_interval=plot_PI_indiv,
                use_ID_for_fig_name=use_ID_for_fig_name)
        elif item == 'VisualPredictiveCheck':
            folder_output = os.path.join(graphic_folder, item)
            if os.path.isdir(folder_output) is False:
                os.mkdir(folder_output)
            plot_vpc(project_folder, folder_output, x_label, y_label )
        elif item == 'ObservationsVsPredictions':
            folder_output = os.path.join(graphic_folder, item)
            if os.path.isdir(folder_output) is False:
                os.mkdir(folder_output)
            plot_obs_vs_pred( project_folder, folder_output )

        elif item == 'PredictionDistribution':
            folder_output = os.path.join(graphic_folder, item)
            if os.path.isdir(folder_output) is False:
                os.mkdir(folder_output)
            plot_prediction_distribution( project_folder, folder_output, x_label, y_label )

        elif item == 'ScatterPlotOfTheResiduals':
            folder_output = os.path.join(graphic_folder, item)
            if os.path.isdir(folder_output) is False:
                os.mkdir(folder_output)
            plot_residuals( project_folder, folder_output, x_label, y_label )
