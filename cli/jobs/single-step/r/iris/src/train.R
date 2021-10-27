library("optparse")
library("carrier")
library(rpart)
source('aml.R')

# Setting Mlflow tracking uri
Sys.setenv(MLFLOW_PYTHON_BIN = "/usr/bin/python")
Sys.setenv(MLFLOW_BIN = "/usr/local/bin/mlflow")

parser <- OptionParser()
parser <- add_option(parser, "--data_folder",
                     type="character", 
                     action="store", 
                     default = "./data", 
                     help="data folder")

args <- parse_args(parser)

print("data folder...\n")
print(args$data_folder)

file_name = file.path(args$data_folder)

print("first 6 rows...\n")
iris <- read.csv(file_name)
print(head(iris))

with(run <- mlflow_start_run(), {
  print("building model...\n")

  model <- rpart(species ~ ., data = iris, method = "class")

  predictor <- crate(~ stats::predict(!!model, .x, method = "class"))
  predicted <- predictor(iris[5:10,])
  print(predicted)

  print("logging model...\n")

  override_log_model(predictor, "model")
})
