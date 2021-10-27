library(optparse)
library("carrier")
source('aml.R')

# Setting Mlflow tracking uri
Sys.setenv(MLFLOW_PYTHON_BIN = "/usr/bin/python")
Sys.setenv(MLFLOW_BIN = "/usr/local/bin/mlflow")

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
  override_log_model(predictor, "model")
})