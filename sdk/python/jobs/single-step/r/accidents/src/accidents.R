library(optparse)
library("carrier")

# Loading azureml_utils.R. This is needed to use AML as MLflow backend tracking store.
source('azureml_utils.R')

# Setting MLflow related env vars
# https://www.mlflow.org/docs/latest/R-api.html#details
Sys.setenv(MLFLOW_BIN=system("which mlflow", intern=TRUE))
Sys.setenv(MLFLOW_PYTHON_BIN=system("which python", intern=TRUE))

options <- list(
  make_option(c("-d", "--data_folder"), default="./data")
)

opt_parser <- OptionParser(option_list = options)
opt <- parse_args(opt_parser)

paste(opt$data_folder)

accidents <- readRDS(file.path(opt$data_folder, "accidents.Rd"))
summary(accidents)

with(run <- mlflow_start_run(), {
  print("Training the model")
  model <- glm(dead ~ dvcat + seatbelt + frontal + sex + ageOFocc + yearVeh + airbag  + occRole, family=binomial, data=accidents)
  summary(model)

  predictor <- crate(~ factor(ifelse(stats::predict(!!model, .x)>0.1, "dead","alive")))
  predictions <- predictor(accidents)
  accuracy <- mean(predictions == accidents$dead)
  
  mlflow_log_metric("accuracy", accuracy)

  print("Logging model")
  mlflow_log_model(predictor, "model")
})