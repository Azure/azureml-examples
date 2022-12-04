# This script is used to develop the training script with mlflow locally
# using local mlflow backend and no

mlflow_python_bin = "/Users/marck/miniforge3/envs/mlflow-r/bin/python"
mlflow_bin = "/Users/marck/miniforge3/envs/mlflow-r/bin/mlflow"

Sys.setenv(MLFLOW_PYTHON_BIN = mlflow_python_bin)
Sys.setenv(MLFLOW_BIN = mlflow_bin)


library(mlflow)
library(optparse)
library(carrier)
library(tidyverse)
library(tsibble)
library(fable) # https://fable.tidyverts.org/index.html
library(janitor) # https://github.com/sfirke/janitor

# Source the aml_utils.R script which is needed to use the MLFlow back end
# with R
# source("azureml_utils.R")

# Set MLFlow related env vars
Sys.setenv(MLFLOW_BIN = system("which mlflow", intern = TRUE))
Sys.setenv(MLFLOW_PYTHON_BIN = system("which python", intern = TRUE))

# parse the command line arguments
parser <- OptionParser()

parser <- add_option(
  parser,
  "--data_file",
  type = "character",
  action = "store",
  default = "data/sales-data.csv"
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

# Load the dataset from the mounted input location.
# This can be done directly with R functions.
# There is no need to use reticulate in a running job

## Modify to read the files from Azure storage using reticulate
# df <- pd$read_csv("azureml://")

file_name <- file.path(args$data_file)

oj_sales_read <- readr::read_csv(file_name) |>
  janitor::clean_names()

# Create a ./outputs directory to store any generated artifact
# Image, model, data, etc. 
# Any files saved to ./outputs will be automatically uploaded 
# to the experiment at the end of the run. For example:

if (!dir.exists(args$output)){
  dir.create(args$output)
}

# Constants (were previously defined in a YAML file in the
# reference example, and ideally can be parametrized

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

# create a plot
forecast_plot <- 
  autoplot(fcast) +
  geom_line(data =sales_for_store_brand |> 
              filter(yr_wk <= yearweek("1992 W18")),
            aes(x = yr_wk, y = logmove)) 


# Create parameters tibble for batch loggine
params_tbl <- tibble(
  key = c("store", "brand", "data_file"),
  value = c(args$store, args$brand, args$data_file)
)

# create a tibble with one row per model and all artifacts
# for that model

all_model_data <- 
  fit |>
  as_tibble() |>
  select(-c(store, brand)) |> 
  pivot_longer(cols = everything(),
               names_to = "model_name",
               values_to = "model_object") |>
  mutate(tidy_coef = map(model_object, tidy)) |>
  inner_join(
    metrics |>
      as_tibble() |> 
      select(-c(store, brand, .type)) |>
      pivot_longer(-c(.model)) |>
      rename(model_name = .model,
             metric = name) |>
      nest(data = c(metric, value)) |> 
      rename(perf_metrics = data) 
  ) |>
  inner_join(
    fcast |>
      as_tibble() |> 
      select(-c(store, brand)) |>
      rename(model_name = .model) |>
      nest(data = -model_name) |>
      rename(fcast = data)
  ) 



# Start the run 
mlflow_start_run()
# Log the store
# Log the brand
# Log the model metrics for each separate model
# Log the model
# Log the plot

# End the rum
mlflow_end_run()



# one more transformation for logging metrics
# metrics are numeric
# need df/tibble with key, value, step and timestamp
metrics_tbl <- 
  metrics |> 
  select(-c(store, brand, .type)) |>  
  pivot_longer(-.model) |> 
  unite("key", c(.model, name)) |> 
  mutate(step = 0,
         timestamp = as.integer(Sys.time()))






all_model_data |> 
  filter(model_name == "arima") |> 
  select(fcast) |> 
  pull()
             

all_model_data |> 
  filter(model_name == "mean") |> 
  unnest(fcast)


tbl_ln <- all_model_data |> 
  filter(model_name == "arima")

#create_log_objects <- function(...)
mdl_obj <- 
  tbl_ln |> 
  pull(model_object)

tdy_coef <- 
  tbl_ln |> 
    pull(tidy_coef) |> 
    magrittr::extract2(1) |> 
    pivot_longer(-term) |> 
  unite("key", c(term, name))

pmet <- tbl_ln |> 
  pull(perf_metrics) |> 
  magrittr::extract2(1) 

metrics_tbl <- bind_rows(
  tdy_coef,
  pmet |> rename(key = metric)) |> 
  mutate(step = 0,
         timestamp = as.integer(Sys.time()))


tg_tbl <- tibble(
  key = "model",
  value = tbl_ln |> pull(model_name)
)

# create model object



forecast(mdl, h =20)

tbl_ln |> 
  pull(tidy_coef)










