import os
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shelve
import shutil
import warnings

from model_definition import Model

# pandas settings
pd.set_option('display.precision', 3)
pd.set_option('display.float_format', lambda x: '%.3g' % x)
# matplotlib settings
plt.rcParams['text.usetex'] = False
plt.rc('font', family='Arial', size = 18)

import seaborn as sns

PI_level = 95 # in % (prediction interval)

def plot_bp_bayes(
    control_stack,
    models_list,
    folder,
    V0,
    x_label,
    y_label):

    '''
    Function to create plots relative to the Bayesian predictions.
    Input:
    - control_stack (pandas series): Pandas Series with all the data
    - models_list (list of strings): list with the models to save
    - folder (string): string with the path of the folder where the plots are saved
    - V0 (scalar): real number with the value of the injected cells
    - x_label (string): string with the name of the x label
    - y_label (string): string with the name of the y label
    '''

    sns.set()
    sns.set_style("white")
    sns.set_context("paper", font_scale=2.5, rc={"lines.linewidth": 2, 'marker.markersize': 10})
    sns.despine(offset=10, trim=False)
    sns.set_style("ticks")

    for ID in np.unique(control_stack.index.get_level_values(0)):
        for model_name in models_list:
            p = [[] for i in range(6)]
            fig = plt.figure(dpi = 200)
            fig_legend = plt.figure(dpi = 200, figsize=(1.5,0.5))
            file_name = os.path.join(folder, model_name, (np.str(ID) + '.out'))
            my_shelf = shelve.open(file_name)
            y = my_shelf['Y_pred']
            time = my_shelf['time']
            data = my_shelf['data']
            ID = my_shelf['ID']
            N  = my_shelf['N']
            my_shelf.close()
            t0 = 0
            Y_med = np.median(y,axis=0)
            indx_V0 = np.argmin(np.abs(time-t0))
            V0_pred = Y_med[indx_V0]

            indx_t0 = np.argmin(np.abs(Y_med-V0))
            t0_pred = time[indx_t0]
            ax = fig.add_subplot(1,1,1)

            t_shift = 0
            indx = np.argmin(np.abs(time-data[ID].index[-N]))
            p[0], = ax.plot(time[indx:]+t_shift, Y_med[indx:], '-', color = 'royalblue', linewidth=4)
            p[1], = ax.plot(time+t_shift, Y_med,':', color = p[0].get_color(), linewidth=4)
            labels     = ['Fit', 'Prediction']
            ax.fill_between(time+t_shift,
                        np.percentile(y, (100-PI_level)/2, axis = 0),
                        np.percentile(y, 100-(100-PI_level)/2, axis = 0),
                        color=p[0].get_color(),#"gray",
                        alpha = 0.1)
            p[2],  = ax.fill(np.nan, np.nan, color = p[0].get_color(), alpha = 0.1)
            labels += ['P.I.']

            p[3], = ax.semilogy(data[ID].index[0:-N]+t_shift, data[ID].values[0:-N],'ko',mfc='none',  markersize = 10)
            p[4], = ax.semilogy(data[ID].index[-N:]+t_shift, data[ID].values[-N:],'ko',  markersize = 10)
            labels += ['Data (predictions)']
            labels += ['Data (fit)']

            ax.axvline(t0, color = 'k')
            ax.axvline(data[ID].index[-N], color = 'k')
            p[5] = ax.axvline(t0_pred, color = 'r', linestyle = '--')
            ax.axhline(V0,color = 'k')

            labels += ['Predicted time']
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

            fig_legend.legend(handles   = p,
                              labels    = labels,
                              loc       = 'center',
                              edgecolor = 'none')
            xt = ax.get_xticks()
            xtl = xt.tolist()
            xtl=list(["$0$","$t_{n-2}$"])

            xt = [t0, data[ID].index[-N]]

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
            if t0_pred < t0:
                t0_pos = t0_pred - 7.5
            else:
                t0_pos = t0_pred + .5
            filename = os.path.join(folder,model_name,np.str(ID)+'.pdf')
            fig.savefig(filename, format = 'pdf', dpi=1000, bbox_inches='tight')
            plt.close('all')

    filelegend = os.path.join(folder,'individual_legend.pdf')
    fig_legend.savefig(filelegend, format='pdf', dpi=1000, bbox_inches='tight')


def swarm_plot(err_t0,
               method,
               models_list,
               folder):
    '''
    function to create swarm plot (with seaborn)
    Input:
    - err_t0 (pandas dataframe): table containing the error between the predicted and the actual value of the initiation time of each individual.
                                 Each column of the table corresponds to a tumor growth model
    - method (string): method used for backward prediction ("bi" or "lm")
    - models_list (list of strings): list with the names of the models that are used
    - folder (string): directory path where swarm plot is saved
    '''
    df = pd.DataFrame()

    max_value = 10

    values = np.array([])
    columns_name = np.array([])
    value_type = np.array([])
    for model in err_t0.columns:
        values = np.concatenate([values,err_t0[model].values])
        columns_name = np.concatenate([columns_name,[model for i in range(len(err_t0[model].values))]])
        value_type = np.concatenate([value_type,[method for i in range(len(err_t0[model].values))]])
    outliers = np.where(np.abs(values)>max_value)[0]
    values[outliers] = np.nan
    df['model'] = columns_name
    df['type'] = value_type
    df['error'] = values

    models_name = list()
    for model_name in models_list:
        model = Model(model_name)
        models_name.append(model.plot_name)

    sns.set()
    sns.set_style("white")
    sns.set_context("paper", font_scale=2.5, rc={"lines.linewidth": 2, 'marker.markersize': 10})
    sns.despine(offset=10, trim=False)
    sns.set_style("ticks")


    fig = plt.figure(dpi=200)

    ax = fig.add_subplot(1,1,1)
    ax.plot(np.arange(-1,5), np.zeros(6), 'k--', linewidth = 1)

    sns.set_context("paper", font_scale=2.5, rc={"lines.linewidth": 0.5, 'marker.markersize': 0.5})
    g = sns.swarmplot(x = 'model', y = 'error', data = df[df['type']==method], hue = 'type',
                    palette=sns.color_palette(['black']),# size = 1,
                    size = 5,
                    edgecolor ='none',
                    ax = ax
    )

    ax.legend('',edgecolor='none', framealpha=0.1)
    ax.plot(np.arange(-1,5), np.zeros(6), 'k--', linewidth = 1)

    sns.set_context("paper", font_scale=2.5, rc={"lines.linewidth": 1, 'marker.markersize': 50})

    ax.set_xticklabels(models_name,rotation=30,horizontalalignment='right')
    ax.set_xlabel('')
    ax.set_ylabel('Relative error')
    sns.despine(right=False, top=False)

    ax.legend('',edgecolor = 'none')
    y_lim = ax.get_ylim()
    ax.set_ylim((-np.abs(y_lim).max(),np.abs(y_lim).max()))
    fig.savefig(os.path.join(folder,'swarmplot_'+method+'.pdf'),
                format = 'pdf', dpi = 1000, bbox_inches='tight')
    plt.close('all')
    fig = plt.figure(dpi = 200)
    fig_legend = plt.figure(dpi = 200, figsize=(1.5,0.5))
    ax = fig.add_subplot(1,1,1)
    p1=ax.errorbar(np.nan, np.nan, np.nan, marker='o', linestyle='', color='k' ,linewidth=2, markersize=8)
    p2,=ax.plot(np.nan,np.nan,marker='o',linestyle='',alpha=0.3,markeredgewidth=0.0)
    plt.close(fig)
    fig_legend.legend(handles   = [p1,p2],
                          labels    = ['Mean $\pm$ std','Individual error'],
                          loc       = 'center',
                          edgecolor = 'none')
    fig_legend.savefig(os.path.join(folder,'swarm_legend.pdf'),
                       format = 'pdf', dpi = 1000, bbox_inches='tight')



def read_results(models_list,
                 folder,
                 n_sets,
                 control_stack,
                 method,
                 V0 = 1):
    '''
    function to read results from the backward predictions workspaces.
    It estimates the accuracy and the precision of the predictions.

    Input:
    - models_list (list of strings): list with the models considered for backward predictions
    - folder (string): directory where workspaces are saved
    - n_sets (int): number of sets used for backward predictions
    - control_stack (pandas series): entire data set
    - method (string): method used to perform backward predictions ("bi" or "lm")
    - V0 (scalar): value relative to the number of injected cells converted in the appropriate unit

    Output:
    err_t0 (pandas Dataframe): table containing the error between the predicted and the actual value of the initiation time of each individual.
                               Each column of the table corresponds to a tumor growth model.
                               Each row corresponds to an individual.
    PI_t0 (pandas Dataframe): table containing the width of the 95% prediction interval of each individual.
                               Each column of the table corresponds to a tumor growth model.
                               Each row corresponds to an individual.
    t0_in_PI (pandas Dataframe): table of bool defining if the actual value of the initiation time of an individual felt in the respective prediction interval.
                                Each column of the table corresponds to a tumor growth model.
                                Each row corresponds to an individual.
    '''
    # initialization of the output variables
    err_t0 = pd.DataFrame()
    PI_t0  = pd.DataFrame()
    t0_in_PI = pd.DataFrame()

    ID_vec = [i for i in np.unique(control_stack.index.get_level_values(0))]

    for model_name in models_list:
        err_t0_list = list()
        PI_list     = list() # this stores the length of the prediction interval
        t0_in_PI_list = list()

        for ID in ID_vec:
            i = 1
            file_name = os.path.join(folder, model_name, (np.str(ID) + '.out'))
            if os.path.isfile(file_name+'.db') is True or os.path.isfile(file_name+'.dat') is True:
                try:
                    file_name = os.path.join(folder, model_name, (np.str(ID) + '.out'))
                    my_shelf = shelve.open(file_name)
                    y        = my_shelf['Y_pred']
                    time     = my_shelf['time']
                    data     = my_shelf['data']
                    ID       = my_shelf['ID']
                    N        = my_shelf['N']
                    t_init   = my_shelf['t_init']

                    t0       = 0
                    t_shift  = data[ID].index[-N]
                    Y_med    = np.median(y,axis = 0)
                    if np.any(np.isnan(t_init)) and method!='lm':# or np.all(Y_med > V0):
                        print(model_name+' - Warning: NaN values in t_init')
                        t_init = t_init[np.where(np.logical_not(np.isnan(t_init)))]
                    t0_pred = np.median(t_init)

                    err_t0_list.append((t0_pred-t0)/np.abs(t0-t_shift))
                    if method == 'lm':
                        t_init = my_shelf['PI_t_init']
                    my_shelf.close()
                    t_perc1 = np.percentile(t_init, (100-PI_level)/2, axis = 0)
                    t_perc2 = np.percentile(t_init, 100-(100-PI_level)/2, axis = 0)
                    if np.any(np.isnan(t_perc2-t_perc1)) or np.abs(t_perc2-t_perc1)>3000:
                        PI_list.append(np.nan)
                        t0_in_PI_list.append(0)
                    else:
                        PI_list.append(np.abs(t_perc2-t_perc1))
                        if np.min([t_perc1, t_perc2])<= t0 and np.max([t_perc1, t_perc2])>= t0:
                            t0_in_PI_list.append(1)
                        else:
                            t0_in_PI_list.append(0)

                except KeyError as error:
                    print(model_name+', '+ np.str(ID))
                    print('Caught this error: ' + repr(error))
                    err_t0_list.append(np.nan)
                    PI_list.append(np.nan)
                    t0_in_PI_list.append(0)
            else:
                err_t0_list.append(np.nan)
                PI_list.append(np.nan)
                t0_in_PI_list.append(0)
        err_t0[model_name] = err_t0_list
        PI_t0[model_name]  = PI_list
        t0_in_PI[model_name]  = t0_in_PI_list
    return err_t0, PI_t0, t0_in_PI

def save_results(folder,
                 err_t0_bi,
                 err_t0_lm,
                 PI_t0_bi,
                 PI_t0_lm
                 ):
    '''
    function to save the results of backward predictions.
    It creates and compiles a latex file containing (i) accuracy and precision of each model and each method and (ii) swarm plots.

    Input:
    - folder (string): name of the directory where the results are saved
    - err_t0_bi (pandas Dataframe): table containing the accuracy of backward predictions of each model using Bayesian inference.
    - err_t0_lm (pandas Dataframe): table containing the accuracy of backward predictions of each model using likelihood maximization.
    - PI_t0_bi (pandas Dataframe): table containing the precision of backward predictions of each model using Bayesian inference.
    - PI_t0_lm (pandas Dataframe): table containing the precision of backward predictions of each model using likelihood maximization.
    '''

    df = pd.DataFrame(columns = list(['Model','Estimation method', 'Error', 'PI']))
    count = 0
    for model_name in err_t0_bi.columns:
        model = Model(model_name)
        temp = list([])
        temp.append(model.plot_name)
        temp.append('Bayesian')
        error = '%.3g (%.3g) ' %(err_t0_bi[model_name].apply(abs).mean()*100, err_t0_bi[model_name].apply(abs).sem()*100)
        temp.append(error)
        PI = '%.3g (%.3g) ' %(PI_t0_bi[model_name].apply(abs).mean(), PI_t0_bi[model_name].apply(abs).sem())
        temp.append(PI)
        df.loc[count] = temp
        count += 1

    for model_name in err_t0_lm.columns:
        model = Model(model_name)
        temp = list([])
        temp.append(model.plot_name)
        temp.append('LM')
        error = '%.3g (%.3g) ' %(err_t0_lm[model_name].apply(abs).mean()*100, err_t0_lm[model_name].apply(abs).sem()*100)
        temp.append(error)
        PI = '%.3g (%.3g) ' %(PI_t0_lm[model_name].apply(abs).mean(), PI_t0_lm[model_name].apply(abs).sem())
        temp.append(PI)
        df.loc[count] = temp
        count += 1

    fid = open(os.path.join(folder,'prediction_results.tex'),'w')
    fid.write('\\begin{table}[H]\n\\centering\n')
    fid.write(df.to_latex(index=False))
    fid.write('\n\n')
    fid.write('\n\\end{table}\n')
    fid.write('\nOutliers: \n')
    fid.write('\nBayesian: \n\n')
    fid.write(err_t0_bi.isna().sum().to_latex(header=False) + '\n')
    fid.write('\nLM: \n\n')
    fid.write(err_t0_lm.isna().sum().to_latex(header=False) + '\n')
    fid.close()

    preamble_file = os.path.join(os.path.dirname(sys.modules[__name__].__file__), 'preamble.tex')
    shutil.copy(preamble_file, folder + '/preamble.tex')
    fid = open(os.path.join(folder,'summary_backward_predictions.tex'),'w')
    fid.write('\\input{preamble.tex}')

    fid.write('\\section{Results}\n\n')
    fid.write('\input{prediction_results.tex}\n ')

    fid.write('\\newpage\n\n')
    fid.write('\\section{Swarm plot}\n\n')
    fid.write('\\begin{figure}[H]\n\\centering\n')
    fid.write(('\\includegraphics[width = 0.7\\textwidth]{swarmplot_bi.pdf}\n'))
    fid.write(('\\caption{Swarm plot of relative errors obtained with bayesian inference}\n'))
    fid.write('\\end{figure}\n\n')
    fid.write('\\begin{figure}[H]\n\\centering\n')
    fid.write(('\\includegraphics[width = 0.7\\textwidth]{swarmplot_lm.pdf}\n'))
    fid.write(('\\caption{Swarm plot of relative errors obtained with likelihood maximization}\n'))
    fid.write('\\end{figure}\n\n')


    fid.write('\n\\end{document}')
    fid.close()

    current_directory = os.getcwd()
    abs_path = os.path.abspath(folder)
    os.chdir(abs_path)
    bash_compile_table = ('pdflatex summary_backward_predictions.tex')
    res = os.system(bash_compile_table)

    if res == 256:
        warnings.warn("Failed in compiling latex file")
    bash_hide_files = ('SetFile -a V *.aux  *.log  *.out')
    os.system(bash_hide_files)
    os.chdir(current_directory)

