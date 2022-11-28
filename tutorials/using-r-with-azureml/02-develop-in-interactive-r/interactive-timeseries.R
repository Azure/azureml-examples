# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

## NOTES

# Code adapted, modified and optimized from the forecasting example 
# https://microsoft.github.io/forecasting/examples/grocery_sales/
# https://github.com/microsoft/forecasting
# accessed 11/14/2022

# The source of the dataset is from the `bayesm` package and has been 
# saved as csv files and provided in the data/ directory.


library(tidyverse)
library(tsibble)
#library(feasts)
library(fable) # https://fable.tidyverts.org/index.html
library(janitor) # https://github.com/sfirke/janitor
library(qs)



## Modify to read the files from Azure storage using reticulate
# df <- pd$read_csv("azureml://")

oj_sales_read <- readr::read_csv("data/yx.csv") |> 
  janitor::clean_names()


library(reticulate)
use_virtualenv("interactive-r")
pd <- import("pandas")

# get the azureml URI from the Datastore UI (is there a way to get this programatically?"

yx_uri <- "azureml://subscriptions/2fcb5846-b560-4f38-8b32-ed6dedcc0a38/resourcegroups/aml/workspaces/marckvaisman-mcaps-nonprod/datastores/marckblob/paths/bayesm-orangejuice/yx.csv"

oj_sales_read <- pd$read_csv(yx_uri)


# Constants (were previously defined in a YAML file in the 
# reference example, and ideally can be parametrized

START_DATE <- as.Date("1989-09-14")  

## Data prep

oj_sales <- oj_sales_read |> 
  # complete the missing combinations
  tidyr::complete(store, brand, week) |> 
  # create the actual week based on start date and # of weeks passed
  mutate(yr_wk = tsibble::yearweek(START_DATE + week * 7)) |> 
  select(-week) |> 
  # convert to tsibble
  as_tsibble(index = yr_wk, key = c(store, brand))

# All stores have the same start week (1990 W25) and end week (1992 W41).
# For training, use 100 weeks (1992 W18)

# The model function in fabletools can fit multiple models 
# out of the box

fit <- oj_sales |> 
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


sg_data <- oj_sales |> filter(store == 2, brand == 1)
sg_model <- fit |> filter(store == 2, brand == 1)
sg_fcast <- fcast |> filter(store == 2, brand == 1)
sg_metrics <- metrics |> filter(store == 2, brand == 1)

sg_fcast |> 
  autoplot(sg_data, level = NULL)

# save intermiediate artifacts
install.packages("qs")
qs::qsavem(fit, metrics, fcast, oj_sales, file = "wip.rda")


# save a model


# https://stackoverflow.com/questions/59721759/extract-model-description-from-a-mable

fit %>% head(2) |> 
  as_tibble() %>%
  gather() %>%
  mutate(model_desc = print(value)) %>%
  select(key, model_desc) %>%
  set_names("model", "model_desc")




