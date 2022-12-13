## AZURE MACHINE LEARNING ONLY

# Source the aml_utils.R script which is needed to use the MLFlow back end
# with R
source("azureml_utils.R")

# Set MLFlow related env vars
Sys.setenv(MLFLOW_BIN = system("which mlflow", intern = TRUE))
Sys.setenv(MLFLOW_PYTHON_BIN = system("which python", intern = TRUE))

## ALL

library(optparse)
library(carrier)
library(tidyverse)
library(tsibble)
library(fable) # https://fable.tidyverts.org/index.html
library(janitor) # https://github.com/sfirke/janitor

# parse the command line arguments
parser <- OptionParser()

parser <- add_option(
  parser,
  "--data_file",
  type = "character",
  action = "store",
  default = "../../data/sales-data.csv"
)

parser <- add_option(
  parser,
  "--output",
  type = "character",
  action = "store",
  default = "./outputs"
)

parser <- add_option(
  parser,
  "--brand",
  type = "double",
  action = "store",
  default = 1
)

parser <- add_option(
  parser,
  "--store",
  type = "double",
  action = "store",
  default = 2
)

args <- parse_args(parser)

# Read the file from the data path provided with the job

file_name <- file.path(args$data_file)

oj_sales_read <- readr::read_csv(file_name) |>
  janitor::clean_names()

# Create a ./outputs directory to store any generated artifacts 
# (images, models, data, etc.) Any files saved to ./outputs will  
# be automatically uploaded # to the experiment at the end of the 
# run. For example:

if (!dir.exists(args$output)){
  dir.create(args$output)
}

# This value is hardcoded

START_DATE <- as.Date("1989-09-14")

## Data prep of the full dataset

oj_sales <- oj_sales_read |> 
  # complete the missing combinations
  tidyr::complete(store, brand, week) |> 
  # create the actual week based on start date and # of weeks passed
  mutate(yr_wk = tsibble::yearweek(START_DATE + week * 7)) |> 
  select(-week) |> 
  # convert to tsibble
  as_tsibble(index = yr_wk, key = c(store, brand)) 
  
## Select the store and brand based on the job parameter  
sales_for_store_brand <- oj_sales  |>
  filter(store == args$store, brand == args$brand)

# All stores have the same start week (1990 W25) and end week (1992 W41).
# For training, use 100 weeks (1992 W18)

# The model function in fabletools can fit multiple models 
# out of the box

fit <- sales_for_store_brand |> 
  filter(yr_wk <= yearweek("1992 W18")) |> 
  model(
    mean = MEAN(logmove),
    naive = NAIVE(logmove),
    drift = RW(logmove ~ drift()),
    arima = ARIMA(logmove ~ pdq() + PDQ(0, 0, 0))
)

# Forecast out 10 weeks 
fcast <- forecast(fit, h = 10)

# Evaluate the metrics for each model (one set of metrics
# per modeltype/store/brand)
metrics <- accuracy(fcast, oj_sales)

# create a plot and save it to output
forecast_plot <- 
  autoplot(fcast) +
  geom_line(data =sales_for_store_brand |> 
              filter(yr_wk <= yearweek("1992 W18")),
            aes(x = yr_wk, y = logmove)) 

ggsave(forecast_plot, 
       filename = "./outputs/forecast-plot.png", 
       units = "px",
       dpi = 100,
       width = 800,
       height = 600)


# create a tibble with one row per model and all re-arranged artifacts
# (tidy info, metrics and forecast), plus tibbles for logging:
# 


all_model_data <-
  fit |>
  #as_tibble() |>
  select(-c(store, brand)) |>
  pivot_longer(cols = everything(),
               names_to = "model_name",
               values_to = "model_object") |>
  mutate(tidy_model = map(model_object, tidy)) |>
  inner_join(metrics |>
               select(-c(store, brand)) |>
               nest(metrics = -c(.model)),
             by = c("model_name" = ".model")) |>
  inner_join(
    fcast |>
      as_tibble() |>
      select(-c(store, brand)) |>
      #      group_by(.model) |>
      group_nest(.model, .key = "prediction"),
    by = c("model_name" = ".model")
  ) |>
  mutate(
    metrics_tbl = map(metrics, function(m) {
    m |>
      select(-c(.type)) |>
      pivot_longer(everything(),
                   names_to = "key") |> 
        mutate(step = 0,
               timestamp = as.integer(Sys.time()))
        
        
  }),
  params_tbl = map(tidy_model, function(tm) {
    tm |> 
      pivot_longer(-term) |> 
      unite("key", c(term, name))
  }),
  tag_tbl = map(model_name, function(n) {
    tibble(key = "model", value = n)
  }))



write_rds(all_model_data, 
          file = "outputs/all-models-tibble.rds")

# one more transformation for logging metrics
# metrics are numeric
# need df/tibble with key, value, step and timestamp

# testing with a single model (arima), need to wrap in function
# to loop and log all four

# extract the models from the tibble and crate them 

mean_ts_pred <- crate(function(x) 
{fabletools::forecast(!!all_model_data$model_object[[1]], h = x)})

naive_ts_pred <- crate(function(x) 
{fabletools::forecast(!!all_model_data$model_object[[2]], h = x)})

drift_ts_pred <- crate(function(x) 
{fabletools::forecast(!!all_model_data$model_object[[3]], h = x)})

arima_ts_pred <- crate(function(x) 
{fabletools::forecast(!!all_model_data$model_object[[4]], h = x)})

# Unlike Python ML models, building and deploying models in R 
# through MLflow requires the model to be packaged before it can be 
# logged as an mlflow model. The crate function in the carrier 
# package is a tool that helps wrap and construct the R model, 
# making it a crated function. This is then passed in to be 
#logged as an mlflow model.

# In this example, the prediction is a set of data points for a 
# time series predicted n-periods after the last period of the training 
# set. It only requires a single number (the number of periods).

# experiment metadata
experiment_tbl <- tibble(
  key = c("store", "brand", "data_file"),
  value = c(args$store, args$brand, args$data_file)
)


# Since we are logging 4 separate models and associated metadata from a
# single script run, we use mlflow with nested runs.

# Start the nested run 
mlflow_start_run()

mlflow_log_param("store", args$store)
mlflow_log_param("brand", args$brand)
mlflow_log_param("data_file", args$data_file)

mlflow_log_artifact(
  path = "./outputs/"
)
# now log nested models and metadata

# log mean
mlflow_start_run(nested = TRUE)

mlflow_log_batch(
  metrics = all_model_data$metrics_tbl[[1]],
  params = all_model_data$params_tbl[[1]],
  tags = all_model_data$params_tbl[[1]]
)

mlflow_log_model(
  model = mean_ts_pred, 
  artifact_path = "model"
)

mlflow_end_run()


mlflow_start_run(nested = TRUE)

mlflow_log_batch(
  metrics = all_model_data$metrics_tbl[[2]],
  params = all_model_data$params_tbl[[2]],
  tags = all_model_data$params_tbl[[2]]
)

mlflow_log_model(
  model = naive_ts_pred, 
  artifact_path = "model"
)

mlflow_end_run()

mlflow_start_run(nested = TRUE)

mlflow_log_batch(
  metrics = all_model_data$metrics_tbl[[3]],
  params = all_model_data$params_tbl[[3]],
  tags = all_model_data$params_tbl[[3]]
)

mlflow_log_model(
  model = drift_ts_pred, 
  artifact_path = "model"
)

#library(purrr) my_tibble |> pull(nested_column) |> pluck(111)

mlflow_end_run()

# log arima - nested
mlflow_start_run(nested = TRUE)

mlflow_log_batch(
  metrics = all_model_data$metrics_tbl[[4]],
  params = all_model_data$params_tbl[[4]],
  tags = all_model_data$params_tbl[[4]]
)

mlflow_log_model(
  model = arima_ts_pred, 
  artifact_path = "model"
)

# end arima
mlflow_end_run()

# end parent run
mlflow_end_run()
