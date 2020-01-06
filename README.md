Program to run predictions of tumor kinetics with bayesian inference and likelihood maximization. Specifically designed to perform backward extrapolation of initiation time.

# Program requirements
These programs must be installed to run the code:
- python 3.6.7
- R 3.5.1
- MonolixSuite2019

# Python packages
- pystan
- scipy
- matplotlib
- numpy
- shelve
- pandas
- statsmodels
- seaborn
- mlx_py*

** to install mlx_py copy the following lines (see mlx_py in modules):
from rpy2 import robjects
robjects.r('install.packages(\'R6\')')
robjects.r('install.packages(\'RJSONIO\')')
robjects.r('install.packages(\'/Applications/MonolixSuite2019R1.app/Contents/Resources/monolixSuite/connectors/lixoftConnectors.tar.gz\', repos = NULL, type=\'source\',, INSTALL_opts =\'--no-multiarch\')') #the path might change.

# To run the code
Run the `code/main.py` file in a terminal (from within the `code` folder) to execute the analysis
```
python3 main.py
```
In the main.py file it is possible to chose the data set (cell line name), the models to perform the analysis, the error model and the number of observations to take into account for the backward prediction.

The program first divides the data set into n subsets that will be used to perform the k-fold cross validation.
Then the following options can be run:
- NLME analysis of the entire data set
- backward prediction with bayesian inference (k-fold cross validation):
  - NLME analysis of the subsets
  - backward predictions with stan
- backward predictions with likelihood maximization

# Results
Results are organized as follows inside the folder folder_results/cell_line_name (specified in the main.py file):
-  global monolix analysis: a summary is created (summary_global_population_analysis.pdf) with the results relative to each model. In the subfolder global_monolix_analysis/ there are files and subfolders with the detailed analysis of the nonlinear mixed effects modeling of each tumor growth equation.
- backward predictions: a summary is created (summary_backward_predictions.pdf) with the relative errors of each model obtained with bayesian inference and likelihood maximization (detailed results and figures are in prediction_summary). Individual prediction plots can be found in backward_prediction/model_name, in case of bayesian inference, and in backward_prediction_lm/model_name, in case of likelihood maximization. The files .db contain the workspaces with the results of backward predictions for each individual.
The folders named with numbers contain the test set and the learning test used for the k-fold cross validation with the Monolix analysis of the learning set. The parameter distributions found are used then as prior distribution to perform bayesian inference.

# Data
The data used in the code are available at:
- breast data measured by volume (breast_vol_data.txt): https://zenodo.org/record/3574531
- lung data measured by volume (lung_vol_data.txt): https://zenodo.org/record/3572401
- breast data measured by fluorescence (breast_fluo_data.txt): https://zenodo.org/record/3593919
# How to add a new data set
It is possible to add a new data set. Data must be saved in a txt file with three columns (ID, Time, Observation) and each row contains the observation of an individual at a certain time. The new data set must be declared in the function import_data in import_data.py.

# How to add a new tumor growth model
Tumor growth models are defined in model_definition.py (for the Monolix analysis and backward predictions using likelihood maximization) and in stan_model_definition.py (for backward predictions using Bayesian inference).
All the models defined in stan_model_definition.py must be defined in model_definition.py.
