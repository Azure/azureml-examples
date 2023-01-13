# plumber.R
# This script will be deployed to a managed endpoint to do the model scoring

# << Get the model directory >>
# When you deploy a model as an online endpoint, AzureML mounts your model
# to your endpoint. Model mounting enables you to deploy new versions of the model without
# having to create a new Docker image. By default, a model registered with the name foo
# and version 1 would be located at the following path inside of your deployed 
# container: var/azureml-app/azureml-models/foo/1

# For example, if you have a directory structure of /azureml-examples/cli/endpoints/online/custom-container on your local # machine, where the model is named half_plus_two:

model_dir <- Sys.getenv("AZUREML_MODEL_DIR")

# << Read the predictor function >>
# This reads the serialized predictor function we stored in a crate
load_model <- readRDS(paste0(model_dir, "/models/crate.bin"))

# Unserialize the crated model to convert it into a function
scoring_function <- unserialize(load_model)

# << Readiness route vs. liveness route >>
# An HTTP server defines paths for both liveness and readiness. A liveness route is used to
# check whether the server is running. A readiness route is used to check whether the 
# server's ready to do work. In machine learning inference, a server could respond 200 OK 
# to a liveness request before loading a model. The server could respond 200 OK to a
# readiness request only after the model has been loaded into memory.

#* Liveness check
#* @get /live
function() {
  "alive"
}

#* Readiness check
#* @get /ready
function() {
  "ready"
}

# << The scoring function >>
# This is the function that is deployed as a web API that will score the model
# Make sure that whatever you are producing as a score can be converted 
# to JSON to be sent back as the API response

#* @param forecast_horizon
#* @post /score
function(forecast_horizon) {
  scoring_function(as.numeric(forecast_horizon)) |> 
    tibble::as_tibble() |> 
    dplyr::transmute(period = as.character(yr_wk),
                     dist = as.character(logmove),
                     forecast = .mean) |> 
    jsonlite::toJSON()
}
