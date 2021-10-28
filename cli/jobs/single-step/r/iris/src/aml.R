library(mlflow)
library(httr)
library(later)

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

# Overriding this due to bug in OSS where name is passed as source
mlflow_create_model_version <- function(name, source, run_id = NULL,
                                        tags = NULL, run_link = NULL,
                                        description = NULL, client = NULL) {
  client <- mlflow:::resolve_client(client)

  response <- mlflow:::mlflow_rest(
    "model-versions",
    "create",
    client = client,
    verb = "POST",
    version = "2.0",
    data = list(
      name = name,
      source = source,
      run_id = run_id,
      run_link = run_link,
      description = description
    )
  )

  return(response$model_version)
}


# Overriding mlflow_log_model since list_artifacts is not implemented
# Once implemented this can be removed.
override_log_model <- function(model, model_name) {
    tryCatch(
        expr = {
            mlflow_log_model(model, model_name)
        },
        error = function(e){ 
            if (grep("API request to endpoint 'artifacts/list' failed with error code 404", toString(e), ignore.case = FALSE, perl = FALSE, fixed = TRUE)){
              print("List artifact failed it is a known issue.")
            }
            else {
              print("Error while logging model")
              print(e)
            }           
        },
        finally = {
                message('log model done')
            }
        )
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
    GET( api_url, timeout(getOption("mlflow.rest.timeout", 60)), req_headers)
}

get_tracking_uri <- function() {
    url <- httr::parse_url(Sys.getenv("MLFLOW_TRACKING_URI"))
    url$query = ""
    url <-httr::build_url(url)
    Sys.setenv(MLFLOW_TRACKING_URI = url)
    url
}

fetch_token_from_aml <- function() {
    tracking_uri <- get_tracking_uri()
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
    }

    if (response$status_code == 200){
        text <- content(response, "text", encoding = "UTF-8")
        json_resp <-jsonlite::fromJSON(text, simplifyVector = FALSE)
        json_resp$token
        print("Setting token")
        Sys.setenv(MLFLOW_TRACKING_TOKEN = json_resp$token)
        print("Setting token done")
    }
}

start_token_refresh <- function() {
    fetch_token_from_aml()
    later::later(start_token_refresh, 1)
}

start_token_refresh()


