# plumber.R

model_dir <- Sys.getenv("AZUREML_MODEL_DIR")

model_names <- tolower(LETTERS[1:3])
load_model <- function(model_name){readRDS(paste0(model_dir,"/models/",model_name,".rds"))}
models <- lapply(model_names, load_model)
names(models) <- model_names

#* Liveness check
#* @get /live
function() {
  "alive"
}

#* Readiness check
#* @get /ready
function() {
  "ready"
}

#* @param model
#* @param x1
#* @param x2 
#* @post /score
function(model, x1, x2) {
  newdata <- data.frame(
    x1=x1, 
    x2=x2
    )
  as.numeric(predict(models[[model]], newdata, type="response"))
}