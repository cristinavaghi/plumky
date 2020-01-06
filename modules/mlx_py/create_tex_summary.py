import numpy as np
import os
import sys
import warnings

# import matplotlib.pyplot and set plot linewidth and font
import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 2
plt.rc('font', family='Arial', size = 12)

# import pandas and set display precision
import pandas as pd
pd.set_option('display.precision', 3)
pd.set_option('display.float_format', lambda x: '%.3g' % x)


import shutil

def statistical_indices_sort_and_save( folder, models, sort_criteria = ['AIC', 'BIC'] ):

    tab = pd.DataFrame(columns = ('model', '-2LL', 'AIC', 'BIC'))
    for item in models:
        tab_loc = pd.DataFrame(columns = ('model', '-2LL', 'AIC', 'BIC'))
        log_like_file = os.path.join(folder, item, 'LogLikelihood','logLikelihood.txt')
        df = pd.read_csv(log_like_file, sep = ',')
        for col in tab.columns:
            if 'importanceSampling' in df:
                tab_loc[col] = df[df['criteria'] == col].loc[:,'importanceSampling'].values
            elif 'linearization' in df:
                tab_loc[col] = df[df['criteria'] == col].loc[:,'linearization'].values
            else:
                warnings.warn('No value of the criterion was found (neither under "importanceSampling" nor "linearization")')
        tab_loc['model'] = item

        tab = tab.append(tab_loc)

    tab = tab.reset_index(drop=True)
    tab = tab.sort_values(by = sort_criteria)

    fid = open(os.path.join(folder,'stat_ind_tab.tex'),'w')
    fid.write('\\begin{table}[H]\n\\centering\n')
    fid.write(tab.to_latex(index=False))
    fid.write('\n\n')
    fid.write('\n\\end{table}\n') #\\caption{Statistical indices.}
    fid.close()

    return tab['model'].values

def VPC_save(folder, models):
    fid = open(os.path.join(folder, 'vpc.tex'),'w')
    fid.write('\\begin{figure}[H]\n\\centering\n')
    legend_file = os.path.join(models[0], 'Graphics', 'VisualPredictiveCheck','legend.pdf')
    fid.write(('\\includegraphics[width = 0.3\\textwidth]{' + legend_file + '}\n'))
    fid.write('\\end{figure}\n\n')

    fid.write('\\begin{figure}[H]\n\\centering\n')
    for item in models:
        vpc_file = os.path.join(item, 'Graphics', 'VisualPredictiveCheck','vpc.pdf')
        fid.write('\\begin{subfigure}[b]{0.45\\textwidth}\n')
        fid.write(('\\includegraphics[width = \\textwidth]{' + vpc_file + '}\n'))
        fid.write(('\\caption{\path{' + item + '}} \n \\end{subfigure}\n'))
    fid.write('\\end{figure}\n\n')
    fid.close()

def prediction_distribution_save(folder, models):
    fid = open(os.path.join(folder, 'prediction_distribution.tex'),'w')
    fid.write('\\begin{figure}[H]\n\\centering\n')
    legend_file = os.path.join(models[0], 'Graphics', 'PredictionDistribution','legend.pdf')
    fid.write(('\\includegraphics[width = 0.3\\textwidth]{' + legend_file + '}\n'))
    fid.write('\\end{figure}\n\n')

    fid.write('\\begin{figure}[H]\n\\centering\n')
    for item in models:
        pred_file = os.path.join(item, 'Graphics', 'PredictionDistribution','prediction_distribution.pdf')
        fid.write('\\begin{subfigure}[b]{0.45\\textwidth}\n')
        fid.write(('\\includegraphics[width = \\textwidth]{' + pred_file + '}\n'))
        fid.write(('\\caption{\path{' + item + '}} \n \\end{subfigure}\n'))
    fid.write('\\end{figure}\n\n')
    fid.close()

def obs_vs_pred_save( folder, models ):
    fid = open(os.path.join(folder ,'obs_vs_pred.tex'),'w')
    fid.write('\\begin{figure}[H]\n\\centering\n')
    legend_file = os.path.join(models[0], 'Graphics', 'ObservationsVsPredictions','legend.pdf')
    fid.write(('\\includegraphics[width = 0.3\\textwidth]{' + legend_file + '}\n'))
    fid.write('\\end{figure}\n\n')

    fid.write('\\begin{figure}[H]\n\\centering\n')
    for item in models:
        obs_file = os.path.join(item, 'Graphics', 'ObservationsVsPredictions','obs_vs_pred.pdf')
        fid.write('\\begin{subfigure}[b]{0.45\\textwidth}\n')
        fid.write(('\\includegraphics[width = \\textwidth]{' + obs_file + '}\n'))
        fid.write(('\\caption{\path{' + item + '}} \n \\end{subfigure}\n'))

    fid.write('\\end{figure}\n\n')
    fid.close()

def scatter_residuals( folder, models ):
    fid = open(os.path.join(folder ,'scatter_residuals.tex'),'w')

    fid.write('\\begin{figure}[H]\n\\centering\n')
    for item in models:
        obs_file = os.path.join(item, 'Graphics', 'ScatterPlotOfTheResiduals','iwres_time.pdf')
        fid.write('\\begin{subfigure}[b]{0.45\\textwidth}\n')
        fid.write(('\\includegraphics[width = \\textwidth]{' + obs_file + '}\n'))
        fid.write(('\\caption{\path{' + item + '}} \n \\end{subfigure}\n'))

    fid.write('\\end{figure}\n\n')
    fid.write('\\begin{figure}[H]\n\\centering\n')
    for item in models:
        obs_file = os.path.join(item, 'Graphics', 'ScatterPlotOfTheResiduals','iwres_obs.pdf')
        fid.write('\\begin{subfigure}[b]{0.45\\textwidth}\n')
        fid.write(('\\includegraphics[width = \\textwidth]{' + obs_file + '}\n'))
        fid.write(('\\caption{\path{' + item + '}} \n \\end{subfigure}\n'))

    fid.write('\\end{figure}\n\n')

    fid.close()

def parameters_save(folder, models ):
    tab     = pd.DataFrame()
    err_tab = pd.DataFrame()
    for item in models:
        tab_loc     = pd.read_csv(os.path.join(folder, item, 'populationParameters.txt'))
        err_tab_loc = pd.DataFrame()
        err_tab_loc = err_tab_loc.append(tab_loc[tab_loc['parameter'] == 'a_'])
        err_tab_loc = err_tab_loc.append(tab_loc[tab_loc['parameter'] == 'a'])
        err_tab_loc = err_tab_loc.append(tab_loc[tab_loc['parameter'] == 'b_'])
        err_tab_loc = err_tab_loc.append(tab_loc[tab_loc['parameter'] == 'b'])
        err_tab_loc = err_tab_loc.append(tab_loc[tab_loc['parameter'] == 'c_'])
        err_tab_loc = err_tab_loc.append(tab_loc[tab_loc['parameter'] == 'c'])
        err_tab_loc.insert(loc = 0, column = 'model', value=np.nan)
        err_tab_loc = err_tab_loc.reset_index(drop = True)
        err_tab_loc.loc[[0],['model']] = item
        err_tab = err_tab.append(err_tab_loc)

        tab_loc = tab_loc[tab_loc['parameter'] != 'a_']
        tab_loc = tab_loc[tab_loc['parameter'] != 'a']
        tab_loc = tab_loc[tab_loc['parameter'] != 'b_']
        tab_loc = tab_loc[tab_loc['parameter'] != 'b']
        tab_loc = tab_loc[tab_loc['parameter'] != 'c_']
        tab_loc = tab_loc[tab_loc['parameter'] != 'c']
        tab_loc.insert(loc = 0, column = 'model', value=np.nan)
        tab_loc.loc[[0],['model']] = item
        tab = tab.append(tab_loc)

    fid = open(os.path.join(folder, 'parameters_tab.tex'),'w')
    fid.write('\\begin{table}[H]\n\\centering\n')
    fid.write(tab.to_latex(index=False, na_rep = ''))
    fid.write('\n\n')
    fid.write('\n\\end{table}\n')#\\caption{Estimated parameters.}
    fid.close()

    fid = open(os.path.join(folder, 'error_model_tab.tex'),'w')
    fid.write('\\begin{table}[H]\n\\centering\n')
    fid.write(err_tab.to_latex(index=False, na_rep = ''))
    fid.write('\n\n')
    fid.write('\n\\end{table}\n')#\\caption{Error model parameters.}
    fid.close()

# main function that creates the global file containing all the results
def save_results(folder, models):
    '''
    the functions compares different Monolix simulations contained in a folder. It generates a .tex file containing the results of the simulations and creates the .pdf file
    Input:
    - folder: string with the path of the folder that contains the different subfolders relative to each simulation
    - models: list of strings containing the names of the subfolders that want to be compared

    The .tex and the .pdf files are generated in 'folder'
    '''
    sort_models = statistical_indices_sort_and_save( folder, models )
    VPC_save(folder, sort_models)
    prediction_distribution_save(folder, sort_models)
    obs_vs_pred_save(folder, sort_models)
    parameters_save(folder, sort_models )
    scatter_residuals( folder, sort_models )

    preamble_file = os.path.join(os.path.dirname(sys.modules[__name__].__file__), 'preamble.tex')
    shutil.copy(preamble_file, folder + '/preamble.tex')
    fid = open(os.path.join(folder, 'summary.tex'),'w')
    fid.write('\\input{preamble.tex}')

    fid.write('\\section{Visual predictive check}\n\n')
    fid.write('\input{vpc.tex}\n ')

    fid.write('\\newpage\n\n')
    fid.write('\\section{Prediction distribution}\n\n')
    fid.write('\input{prediction_distribution.tex}\n ')

    fid.write('\\newpage\n\n')
    fid.write('\\section{Observations vs predictions}\n\n')
    fid.write('\input{obs_vs_pred.tex}\n ')

    fid.write('\\newpage\n\n')
    fid.write('\\section{Scatter plot of the residuals}\n\n')
    fid.write('\input{scatter_residuals.tex}\n ')

    fid.write('\\newpage\n\n')
    fid.write('\\section{AIC, BIC}\n\n')
    fid.write('\input{stat_ind_tab.tex}\n ')

    # fid.write('\\newpage\n\n')
    # fid.write('\\section{Likelihood ratio test}\n\n')
    # fid.write('\input{likelihood_tab.tex}\n ')
    #
    fid.write('\\newpage\n\n')
    fid.write('\\section{Estimated parameters}\n\n')
    fid.write('\input{parameters_tab.tex}\n ')

    fid.write('\\newpage\n\n')
    fid.write('\\section{Error model parameters}\n\n')
    fid.write('\input{error_model_tab.tex}\n ')

    fid.write('\n\\end{document}')
    fid.close()

    current_directory = os.getcwd()
    abs_path = os.path.abspath(folder)
    os.chdir(abs_path)
    bash_compile_table = ('pdflatex -shell-escape -interaction=nonstopmode -file-line-error summary.tex  | grep -i ".*:[0-9]*:.*"')
    res = os.system(bash_compile_table)

    if res == 256:
        warnings.warn("Failed in compiling latex file")
    bash_hide_files = ('SetFile -a V *.aux  *.log  *.out')
    os.system(bash_hide_files)
    os.chdir(current_directory)

