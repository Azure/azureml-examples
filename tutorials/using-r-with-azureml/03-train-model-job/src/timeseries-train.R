# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

## NOTES

# Code adapted, modified and optimized from the forecasting example
# https://microsoft.github.io/forecasting/examples/grocery_sales/
# https://github.com/microsoft/forecasting
# accessed 11/14/2022

# Make sure that packages you are going to use are installed in the
# Docker image, or use pacman (which needs to be installed in the Docker image)
# to check for installed packages and install missing ones at runtime


# Parameters
# store, brand, dataset

library(optparse)
library(carrier)
library(tidyverse)
library(tsibble)
library(fable) # https://fable.tidyverts.org/index.html
library(janitor) # https://github.com/sfirke/janitor

# Source the aml_utils.R script which is needed to use the MLFlow back end
# with R
source("azureml_utils.R")

# Set MLFlow related env vars
Sys.setenv(MLFLOW_BIN = system("which mlflow", intern = TRUE))
Sys.setenv(MLFLOW_PYTHON_BIN = system("which python", intern = TRUE))

# parse the command line arguments
parser <- OptionParser()

parser <- add_option(
  parser,
  "--data_file",
  type = "character",
  action = "store"
)

parser <- add_option(
  parser,
  "--brand",
  type = "double",
  action = "store"
)

parser <- add_option(
  parser,
  "--store",
  type = "double",
  action = "store"
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


# Log the store
# Log the brand
# Log the model metrics for each separate model
# Log the model
# Log the plot

# Add car


sg_data <- oj_sales |> filter(store == 2, brand == 1)
sg_model <- fit |> filter(store == 2, brand == 1)
sg_fcast <- fcast |> filter(store == 2, brand == 1)
sg_metrics <- metrics |> filter(store == 2, brand == 1)

sg_fcast |> 
  autoplot(sg_data, level = NULL)

# save a model


# https://stackoverflow.com/questions/59721759/extract-model-description-from-a-mable

fit %>% head(2) |> 
  as_tibble() %>%
  gather() %>%
  mutate(model_desc = print(value)) %>%
  select(key, model_desc) %>%
  set_names("model", "model_desc")




