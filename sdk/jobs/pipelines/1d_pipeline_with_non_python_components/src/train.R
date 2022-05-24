library("optparse")
library("carrier")
library(rpart)

# Loading aml_utils.R. This is needed to use AML as MLflow backend tracking store.
source('azureml_utils.R')

# Setting MLflow related env vars
# https://www.mlflow.org/docs/latest/R-api.html#details
Sys.setenv(MLFLOW_BIN=system("which mlflow", intern=TRUE))
Sys.setenv(MLFLOW_PYTHON_BIN=system("which python", intern=TRUE))

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

  mlflow_log_model(predictor, "model")
})