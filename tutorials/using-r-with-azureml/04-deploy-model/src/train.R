# train.R
library(rpart)
library(carrier)

# set working directory to the project
setwd("./azureml-deploy-r")

# train a model on the iris data
model <- rpart(Species ~ ., data = iris, method = "class")

# create a crate
predictor <- crate(~ stats::predict(!!model, .x, method = "class"))

# save the crate to an rds file
saveRDS(predictor, file="./models/iris-model.rds")