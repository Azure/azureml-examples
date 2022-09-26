# Azure ML utility to enable usage of the MLFlow R API for tracking with Azure Machine Learning (Azure ML). This utility does the following:
# 1. Understands Azure ML MLflow tracking url by extending OSS MLflow R client.
# 2. Manages Azure ML Token refresh for remote runs (runs that execute in Azure Machine Learning). It uses tcktk2 R libraray to schedule token refresh.
#    Token refresh interval can be controlled by setting the environment variable MLFLOW_AML_TOKEN_REFRESH_INTERVAL and defaults to 30 seconds.

library(mlflow)
library(httr)
library(later)
library(tcltk2)

new_mlflow_client.mlflow_azureml <- function(tracking_uri) {
  host <- paste("https", tracking_uri$path, sep = "://")
  get_host_creds <- function () {
    mlflow:::new_mlflow_host_creds(
      host = host,
      token = Sys.getenv("MLFLOW_TRACKING_TOKEN"),
      username = Sys.getenv("MLFLOW_TRACKING_USERNAME", NA),
      password = Sys.getenv("MLFLOW_TRACKING_PASSWORD", NA),
      insecure = Sys.getenv("MLFLOW_TRACKING_INSECURE", NA)
    )
  }
  cli_env <- function() {
    creds <- get_host_creds()
    res <- list(
      MLFLOW_TRACKING_USERNAME = creds$username,
      MLFLOW_TRACKING_PASSWORD = creds$password,
      MLFLOW_TRACKING_TOKEN = creds$token,
      MLFLOW_TRACKING_INSECURE = creds$insecure
    )
    res[!is.na(res)]
  }
  mlflow:::new_mlflow_client_impl(get_host_creds, cli_env, class = "mlflow_azureml_client")
}

get_auth_header <- function() {
    headers <- list()
    auth_token <- Sys.getenv("MLFLOW_TRACKING_TOKEN")
    auth_header <- paste("Bearer", auth_token, sep = " ")
    headers$Authorization <- auth_header
    headers
}

get_token <- function(host, exp_id, run_id) {
    req_headers <- do.call(httr::add_headers, get_auth_header())
    token_host <- gsub("mlflow/v1.0","history/v1.0", host)
    token_host <- gsub("azureml://","https://", token_host)
    api_url <- paste0(token_host, "/experimentids/", exp_id, "/runs/", run_id, "/token")
    GET( api_url, timeout(getOption("mlflow.rest.timeout", 30)), req_headers)
}


fetch_token_from_aml <- function() {
    message("Refreshing token")
    tracking_uri <- Sys.getenv("MLFLOW_TRACKING_URI")
    exp_id <- Sys.getenv("MLFLOW_EXPERIMENT_ID")
    run_id <- Sys.getenv("MLFLOW_RUN_ID")
    sleep_for <- 1
    time_left <- 30
    response <- get_token(tracking_uri, exp_id, run_id)
    while (response$status_code == 429 && time_left > 0) {
        time_left <- time_left - sleep_for
        warning(paste("Request returned with status code 429 (Rate limit exceeded). Retrying after ",
                    sleep_for, " seconds. Will continue to retry 429s for up to ", time_left,
                    " second.", sep = ""))
        Sys.sleep(sleep_for)
        sleep_for <- min(time_left, sleep_for * 2)
        response <- get_token(tracking_uri, exp_id)
    }

    if (response$status_code != 200){
        error_response = paste("Error fetching token will try again after sometime: ", str(response), sep = " ")
        warning(error_response)
    }

    if (response$status_code == 200){
        text <- content(response, "text", encoding = "UTF-8")
        json_resp <-jsonlite::fromJSON(text, simplifyVector = FALSE)
        json_resp$token
        Sys.setenv(MLFLOW_TRACKING_TOKEN = json_resp$token)
        message("Refreshing token done")
    }
}

clean_tracking_uri <- function() {
    tracking_uri <- httr::parse_url(Sys.getenv("MLFLOW_TRACKING_URI"))
    tracking_uri$query = ""
    tracking_uri <-httr::build_url(tracking_uri)
    Sys.setenv(MLFLOW_TRACKING_URI = tracking_uri)
}

clean_tracking_uri()
tcltk2::tclTaskSchedule(as.integer(Sys.getenv("MLFLOW_TOKEN_REFRESH_INTERVAL_SECONDS", 30))*1000, fetch_token_from_aml(), id = "fetch_token_from_aml", redo = TRUE)