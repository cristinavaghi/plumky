
#data_path = "/Users/cristina/Documents/PhD/materiale/shared_folders/nanoparticles_pharmacometrics/python_codes/cell_doubling_time/monolix_analysis/MDA-MB-231/MDA-MB-231.txt"
#project_path = "/Users/cristina/Documents/PhD/materiale/shared_folders/nanoparticles_pharmacometrics/python_codes/cell_doubling_time/monolix_analysis/MDA-MB-231/prova_monolix.mlxtran"
#model_path = "/Users/cristina/Documents/PhD/materiale/shared_folders/nanoparticles_pharmacometrics/python_codes/cell_doubling_time/monolix_analysis/MDA-MB-231/exponential/Exp.txt"  

header_types = c("ID","TIME","OBSERVATION")
observation_types = "continuous"

#parameter_names = c("V_pop", "a_pop")
#initial_values = c(1,0.015)
#error_model = "constant"

print(parameter_names)
print(project_path)
print(model_path)

header_types = c("ID","TIME","OBSERVATION")
observation_types = "continuous"


scenario_tasks = c(populationParameterEstimation   = T,
                   conditionalModeEstimation       = F,
                   conditionalDistributionSampling = F,
                   standardErrorEstimation         = T,
                   logLikelihoodEstimation         = T,
                   plots                           = T
                   )


plot_list = c("outputplot","indfits","obspred","parameterdistribution",
            "covariatemodeldiagnosis","randomeffects","covariancemodeldiagnosis",
            "saemresults","vpc","categorizedoutput","predictiondistribution")


source("Monolix_R_API.R")

run_monolix( data_path, model_path, project_path, 
           	 header_types, observation_types, 
          	 parameter_names, initial_values, 
             error_model, scenario_tasks, plot_list )