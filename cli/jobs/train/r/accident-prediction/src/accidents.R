library(optparse)

options <- list(
  make_option(c("-d", "--data_folder"), default="./data")
)

opt_parser <- OptionParser(option_list = options)
opt <- parse_args(opt_parser)

paste(opt$data_folder)

accidents <- readRDS(file.path(opt$data_folder, "accidents.Rd"))
summary(accidents)

mod <- glm(dead ~ dvcat + seatbelt + frontal + sex + ageOFocc + yearVeh + airbag  + occRole, family=binomial, data=accidents)
summary(mod)
predictions <- factor(ifelse(predict(mod)>0.1, "dead","alive"))
accuracy <- mean(predictions == accidents$dead)

output_dir = "outputs"
if (!dir.exists(output_dir)){
  dir.create(output_dir)
}
saveRDS(mod, file = "./outputs/model.rds")
message("Model saved")