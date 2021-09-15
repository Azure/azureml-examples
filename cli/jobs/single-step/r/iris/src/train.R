library("optparse")
parser <- OptionParser()
parser <- add_option(parser, "--data_folder",
                     type="character", 
                     action="store", 
                     default = "./data", 
                     help="data folder")

args <- parse_args(parser)

cat("data folder...\n")
print(args$data_folder)

file_name = file.path(args$data_folder)

cat("first 6 rows...\n")
x <- read.csv(file_name)
print(head(x))

cat("building model...\n")
model <- rpart::rpart(species~., data=x)

cat("saving model...\n")
output_dir = "outputs"
if (!dir.exists(output_dir)){
  dir.create(output_dir)
}
saveRDS(model, file = "./outputs/model.rds")
