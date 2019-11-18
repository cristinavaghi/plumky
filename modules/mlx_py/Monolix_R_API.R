##################################################################################################################################################
# LIBRARIES
library(lixoftConnectors) # this library enables the connection with the Monolix program.
					   # references: http://monolix.lixoft.com/monolix-api/
					   # To install it (on mac OS):
					   # first install the version of Monolix MonolixSuite2018R1
					   # write the following R command:  install.packages("/Applications/MonolixSuite2018R1.app/Contents/Resources/mlxsuite/mlxConnectors/R/MlxConnectors.tar.gz", repos = NULL, type='source')
##################################################################################################################################################
# INTERNAL METHODS
#----------------------------------------------------------------------------------

check_file <- function( filepath )
{
	# check if the string is a file
	if (!file.exists(filepath))
	{
    	stop(paste(filepath, ' is not a file'), call. = F)
	}
	return
}
#----------------------------------------------------------------------------------
# set initial conditions
set_initial_conditions <- function( parameter_names_to_set,
									initial_values,
									fixed_params,
									params_fixed_effects,
									covariates,
									correlation)
{
	# the function modifies the initial values of the parameters defined in parameter_names_to_set.
	# INPUT:
	# - parameter_names_to_set: vector with the names of the parameters which initial values have to be set (e.g., parameter_names_to_set = c("V0_pop","a_pop") )
	# - initial_values: vector of float with the initial values of the parameters defined in parameter_names_to_set (e.g., initial_values = c(1,0.015) ). Note that parameter_names_to_set and initial_values must have the same dimension

	if (length(parameter_names_to_set)!=length(initial_values))
	{
		stop('parameter_names and initial_values have different length')
	}

	if(length(covariates) != 0L)
	{
		setCovariateModel(covariates)
	}

	# set the parameters which have fixed effects
    if(length(params_fixed_effects)!=0)
    {
        tab_indiv_model = getIndividualParameterModel()
        for (i in 1:length(params_fixed_effects))
        {
            tab_indiv_model$variability$id[params_fixed_effects[i]] = FALSE
			tab_indiv_model$distribution[params_fixed_effects[i]] = "normal" # set fixed effects normally distributed in order to have the possibility to set negative values
        }
        setIndividualParameterVariability(tab_indiv_model$variability$id)
		setIndividualParameterDistribution(tab_indiv_model$distribution)
    }


	tab = getPopulationParameterInformation()		# this function returns the table with the names, the initial values (by default they are equal to one) and the method of the model parameters
	for (i in 1:length(parameter_names_to_set)){
		tab[tab["name"]==parameter_names_to_set[i],"initialValue"] = initial_values[i]
	}

	if(length(fixed_params)!=0)
		for (i in 1:length(fixed_params))
		{
			tab[tab["name"]== fixed_params[i],"method"] = "FIXED"
		}
	setPopulationParameterInformation(tab) # this function modifies the parameter settings in the Monolix framework
	if (length(correlation) != 0L)
	{
		setCorrelationBlocks(id = list(correlation) )
	}
	return
}
#----------------------------------------------------------------------------------
# set error model
set_error_model <- function(error_model )
{
	if(is.null(error_model))
	{
		error_model = "constant"
	}
	setErrorModel(Observation = error_model) #Monolix function that sets the error model
	return
}
#----------------------------------------------------------------------------------
# set scenario
set_scenario <- function(scenario_tasks, plot_list, linearization)
{
	scenario = getScenario() #Monolix function to get current scenario

	if (is.null(scenario_tasks)) # default values
	{
		scenario_tasks = c( populationParameterEstimation   = T,
                   			conditionalModeEstimation       = T,
                			conditionalDistributionSampling = F,
               			    standardErrorEstimation         = T,
              			    logLikelihoodEstimation         = T,
                   			plots                           = T
                   			)
	}

	if (is.null(plot_list)) # default values
	{
		plot_list = c("indfits","obspred","residualsscatter",
              "covariancemodeldiagnosis","residualsdistribution",
              "vpc","predictiondistribution")
	}

	# tasks
	scenario$tasks = scenario_tasks

	# linearization
	scenario$linearization = linearization

	# plot list
	scenario$plotList = plot_list

	# set scenario
	setScenario(scenario) # Monolix function to set the scenario
	return
}
##################################################################################################################################################
# FUNCTION
run_monolix <- function(data_path,
						model_path,
						project_path,
						parameter_names_to_set = NULL,
						initial_values         = NULL,
						fixed_params           = NULL,
						params_fixed_effects   = NULL,
						covariates             = NULL,
						correlation            = NULL,
						error_model            = "constant",
						header_types           = c("ID","TIME","OBSERVATION"),
						observation_type       = "continuous",
						scenario_tasks         = NULL,
						plot_list              = NULL,
						linearization          = FALSE )
{

	# R function to run the API version of Monolix. Reference: http://monolix.lixoft.com/monolix-api/
	# Settings: linearization = TRUE
	# the function returns the results of the SAEM algorithm and saves in the folder defined
	# in project_path the project (file .mlxtran) and its results.
	# Moreover, it exports the graphic data (ChartsData) of the plots defined in plot_list. It does not save any plot.

	# INPUT:
	# - data_path: string with path of the file that contains the data.
	# - model_path: string with the path of the file that contains the model definition
	# - project_file: string with the path of the folder where the project is saved
	# - header_types: vector with the definition of the headers of the data (e.g., header_types = c("ID","TIME","OBSERVATION"))
	# - observation_type: string with the definition of the type of the observation (e.g., observation_type = "continuous")
	# - parameter_names_to_set: vector with the names of the parameters which initial values have to be set (e.g., parameter_names_to_set = c("V0_pop","a_pop") )
	# - initial_values: vector of float with the initial values of the parameters defined in parameter_names_to_set (e.g., initial_values = c(1,0.015) ). Note that parameter_names_to_set and initial_values must have the same dimension
	# - fixed_params: list of parameter names that are fixed
	# - params_fixed_effects: list of parameters with fixed effects
	# - VI: bool variable (default: FALSE). If it is TRUE, the program considers V0 as the first measurement of the individuals. In this case, the file data_path must contain four columns {ID, Time, Observation, VI}. The last one contains the first measurement of each animal. The program assumes that VI is the measurement at time t=0.
	# - error_model: string defining the error model (e.g. error_model = "constant" or error_model = "combined1")
	# - scenario_tasks: vector containing the tasks for the scenario. (e.g. scenario_tasks = c(populationParameterEstimation = T,conditionalModeEstimation = F, conditionalDistributionSampling = F, standardErrorEstimation = T, logLikelihoodEstimation = T, plots = T))
	# - plot_list: vector of strings containing the plots that are saved
	# - linearization: TRUE or FALSE

	# OUTPUT:
	# list containing
	# - pop_params: named vector containing the estimated values of each one of the population parameters
	# - log_l: estimated log-likelihood values
	# - std_err: estimated standard errors
	# - cor_params: estimated correlation of the parameters


	# check on the input files
    check_file(data_path)
  	check_file(model_path)

  	# ----
  	# inizialization of the connector
  	initializeLixoftConnectors(software = "monolix")

  	# ----
  	# definition of the new project

  	newProject(data      = list(dataFile    = data_path,
   	                       headerTypes      = header_types),
               modelFile = model_path
               )
	setData(dataFile         = data_path,
            headerTypes      = header_types,
            observationTypes = observation_type)

	# ----
	# parameter settings: initial values
	set_initial_conditions(parameter_names_to_set,
						   initial_values,
						   fixed_params,
						   params_fixed_effects,
						   covariates,
						   correlation )
	set_error_model(error_model)
	# ----
	# set scenario: output of the code (e.g. tasks, linearization, plots list)
	set_scenario(scenario_tasks, plot_list, linearization)


	# ----
	# save project in the directory project_path
	saveProject(project_path)

	# ----
	# ----
	# RUN MONOLIX
	runScenario()

	computeChartsData()

	# ----
	# ----
	return
}
