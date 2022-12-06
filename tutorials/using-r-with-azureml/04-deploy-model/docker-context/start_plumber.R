# ./docker-context/start_plumber.R

entry_script_path <- paste0(Sys.getenv('AML_APP_ROOT'),'/', Sys.getenv('AZUREML_ENTRY_SCRIPT'))

pr <- plumber::plumb(entry_script_path)

args <- list(host = '0.0.0.0', port = 8000); 

if (packageVersion('plumber') >= '1.0.0') {
  pr$setDocs(TRUE)
} else { 
  args$swagger <- TRUE 
} 

do.call(pr$run, args)