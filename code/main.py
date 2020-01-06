
cell_line_name = 'MDA-MB-231_fluorescence' # it defines the data set
error_model    = 'proportional' # error model for the monolix analysis (e = a + b*f(t,theta))
folder_results = 'results' # name of the folder where the results are saved:
                            # In this folder the following folders are created:
                            # - global_monolix_analysis: where the NLME analysis on the entire dataset is performed
                            # - 1,...,n: n folders where for each subset the Monolix analysis is performed and where
                            #   the population parameter distributions are saved for the backward predictions with bayesian inference
N              = 3 # number of observations to take into account for the backward predictions

run_global_monolix_analysis = 0 # to run global monolix analysis
run_k_fold_monolix          = 0 # to run monolix for the k-fold cross validation
clean_monolix_workspace     = 0 # to clean the folder where monolix results are saved (in case of k-fold cross validation; only population parameters remain)
run_k_fold_predictions_bi   = 0 # to run backward predictions using bayesian inference
run_predictions_lm          = 0 # to run backward predictions using likelihood maximization
create_graphics             = 1 # to create summary of backward predictions
clean_backward_workspace    = 1 # to clean the folder with the results of backward predictions (only individual plots remain)



models_list = list([
                    # 'exponential',
                    # 'logistic',
                    # 'gompertz',
                    'reduced_gompertz'
                    ])


exec(open('launch_code.py').read())

