## ADDITIONAL MONOLIX FUNCTIONS
import pandas as pd
pd.set_option('display.precision', 3)
pd.set_option('display.float_format', lambda x: '%.3g' % x)
import numpy as np
import os
from linear_regression import *
import mlx_py as mlx
import matplotlib.pyplot as plt
from model_definition import Model
plt.rcParams['lines.linewidth'] = 2
plt.rc('font', family='Arial', size = 20)
plt.rcParams['text.usetex'] = False

def monolix_computation(data_path,
                        models_list,
                        monolix_folder,
                        V0,
                        Vc,
                        error_model,
                        x_label = '',
                        y_label = '',
                        lambda_alpha = []):

    '''
    it performes Monolix analysis on the data set defined by data_path and with the models defined in models_list
    Input:
    - data_path (string): path of the folder with the data set
    - models_list (list): list with the names of the models that are used in the monolix analysis
    - k_folder (string): path of the folder where the results are saved
    - V0 (float): value of the initial volume
    - Vc (float): value of Vc (Gompertz model)
    - error_model (string): error model type (e.g. constant, proportional or combined1)
    - x_label (string): name of the x label
    - y_label (string): name of the y label
    - lambda_alpha (scalar): real value of the in vitro proliferation rate
    '''
    for model_name in models_list:
        correlation = []
        covariates = {}
        if os.path.isdir(monolix_folder) is False:
            os.mkdir(monolix_folder)
        project_path = os.path.join(monolix_folder, model_name + '.mlxtran')


        model = Model(model_name, V0=V0, Vc=Vc)


        if model_name == 'gompertz':
            correlation = ['alpha0','beta']

        mlx.launch_monolix(data_path,
                           'monolix_models/'+model_name+'.txt',
                           project_path,
                           model.param_names + model.fixed_values,
                           model.all_initial_values,
                           model.fixed_values,
                           model.fixed_effects,
                           covariates = covariates,
                           correlated_parameters = correlation,
                           error_model = error_model,
                           header_types = list(["ID", "TIME", "OBSERVATION"]),
                           linearization = False)

        if x_label != '':
            mlx.save_monolix_graphics( os.path.join(monolix_folder, model_name), x_label, y_label, plot_PI_indiv=False )

            if model_name == 'reduced_gompertz':
                params = read_pop_parameters(os.path.join(monolix_folder, model_name))
                k = params[params['parameter']=='k_pop']['value'].values[0]
                model = Model(model_name, V0=V0, Vc=Vc, k = k)

            plot_individual_fits_time0(model, os.path.join(monolix_folder, model_name), x_label, y_label)

            if model_name == 'gompertz':
                plot_correlation(monolix_folder, lambda_alpha)

    if x_label != '':
        mlx.save_results(monolix_folder, models_list)

def read_pop_parameters(folder):
    '''
    Function to read the error model parameters from the Monolix analysis
    Input:
        folder (string): path of the folder containing the populationParameters.txt file
    Output:
        df: pandas table with the value of the parameters

    '''
    file = os.path.join(folder, 'populationParameters.txt')
    df = pd.read_csv(file, sep = ',')
    return df

def read_error_model_parameters(folder):
    '''
    Function to read the error model parameters from the Monolix analysis
    Input:
        folder (string): path of the folder containing the populationParameters.txt file
    Output:
        a, b, c (scalars): values of the error model parameter.
        The error model is defined as e = (a + b*f(t,theta)^c)*eps, where f is the model and eps follows a normal distribution
    '''
    err_tab = pd.DataFrame()
    tab_loc     = pd.read_csv(os.path.join(folder, 'populationParameters.txt'))
    err_tab = err_tab.append(tab_loc[tab_loc['parameter'] == 'a_'])
    err_tab = err_tab.append(tab_loc[tab_loc['parameter'] == 'a'])
    err_tab = err_tab.append(tab_loc[tab_loc['parameter'] == 'b_'])
    err_tab = err_tab.append(tab_loc[tab_loc['parameter'] == 'b'])
    err_tab = err_tab.append(tab_loc[tab_loc['parameter'] == 'c_'])
    err_tab = err_tab.append(tab_loc[tab_loc['parameter'] == 'c'])
    #err_tab.insert(loc = 0, column = 'model', value=np.nan)
    err_tab = err_tab.reset_index(drop = True)
    #err_tab.loc[[0],['model']] = item
    if np.any(['a' in item for item in err_tab['parameter'].values]):
        a = err_tab['value'].values[np.where(['a' in item for item in err_tab['parameter'].values])[0][0]]
    else:
        a = 0
    if np.any(['b' in item for item in err_tab['parameter'].values]):
        b = err_tab['value'].values[np.where(['b' in item for item in err_tab['parameter'].values])[0][0]]
    else:
        b = 0
    if np.any(['c' in item for item in err_tab['parameter'].values]):
        c = err_tab['value'].values[np.where(['c' in item for item in err_tab['parameter'].values])[0][0]]
    else:
        c = 1
    return a, b, c

def plot_individual_fits_time0(
    model,
    folder_input,
    x_label,
    y_label
    ):

    file_path = os.path.join(folder_input, 'individualParameters', 'estimatedIndividualParameters.txt')
    df = pd.read_csv(file_path, sep = ',')

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

    for ID in ID_vec:
        params = list()
        for item in model.param_names:
            params.append(df[df['id']==ID][item+'_mode'].values[0])
        # if model.model_name == 'gompertz_m':
        #     params = list([df[df['id']==ID]['beta_mode'].values[0]])

        time_obs   = df_obs[df_obs['ID']   == ID].loc[: ,'time'].values
        obs        = df_obs[df_obs['ID']   == ID].loc[: , obs_name].values

        time = np.linspace(0,time_obs.max(),100)
        fig = plt.figure(dpi = 100)
        plt.plot(time,model.function(time, *params), color = 'royalblue')
        plt.plot(time_obs, obs,'ko')

        plt.ylabel(y_label)
        plt.xlabel(x_label)

        folder_output = os.path.join(folder_input, 'Graphics', 'IndividualFits0')
        if os.path.isdir(folder_output) is False:
            os.mkdir(folder_output)
        file_output = os.path.join(folder_output, (np.str(ID) + '.pdf'))
        fig.savefig(file_output, format='pdf', dpi=1000, bbox_inches='tight')
        plt.close(fig)


def plot_regression_line(x, y, b, Rsquare = 1, pvalmin=1e-22):
    plt.rc('font', family='Arial', size = 16)


    # plotting the actual points as scatter plot
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(1,1,1)
    ax.plot(x, y, 'ko', markersize = 5, label = 'Individual parameters')

    # predicted response vector
    y_pred = b[0] + b[1]*x

    # plotting the regression line
    ax.plot(x, y_pred, color = "gray", label = 'fit')

    # putting labels
    ax.set_ylabel('$\\alpha$ (day$^{-1}$)')
    ax.set_xlabel('$\\beta$ (day$^{-1}$)')

    textstr = '\n'.join((
        r'$R^2 = %.3g$' % (Rsquare, ),
        #r'$p$-value < %.3g' % (pvalmin, )))
        r'$p$-value < $10^{-5}$'  ))
    # ax.text(np.min(x)*1.01, np.max(y)*0.7, textstr)
    fig_text = plt.figure(dpi = 200, figsize=(1.5,0.5))
    fig_text.text(0, 0, textstr)
    return fig, ax, fig_text

def plot_correlation(
    folder,
    lambda_alpha):

    file_path = os.path.join(folder,'gompertz','individualParameters','estimatedIndividualParameters.txt')
    df = pd.read_csv(file_path, sep = ',')
    alpha = df['alpha0_mode']
    beta = df['beta_mode']

    lr = linear_regression(beta.values, alpha.values, 1)
    fig, ax, fig_text = plot_regression_line(beta, alpha, lr.params, np.min([lr.rsquared, 0.99]), np.max([1e-5,np.min(lr.pvalues)]))
    ax.plot([beta.min(),beta.max()], lambda_alpha*np.ones(2),'k:', label = '$\lambda$ = '+np.str(lambda_alpha))
    ax.legend(loc=4, fontsize=13)
    figname=os.path.join(folder, 'correlation.pdf')
    fig.savefig(figname, dpi = 1000, format = 'pdf', bbox_inches='tight')
    fig_text.savefig(os.path.join(folder, 'correlation_box.pdf'), dpi = 1000, format = 'pdf', bbox_inches='tight')

