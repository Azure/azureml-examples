# plumber.R

accident_model <- readRDS("model.rds")

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

#* Returns likelihood of fatality given information about a car accident
#* @param dvcat, velocity, possible values "1-9km/h" "10-24"   "25-39"   "40-54"   "55+"  
#* @param seatbelt, possible values "none" "belted"  
#* @param frontal, possible values "notfrontal" "frontal"
#* @param sex, possible values "f" "m"
#* @param ageOfocc, possible values age in years, 16-97
#* @param yearVeh, year of vehicle, possible values 1955-2003
#* @param airbag, was an airbag present? possible values "none" "airbag"
#* @param occRole, possible values "driver" "pass" 
#* @post /score
function(dvcat, seatbelt, frontal, sex, ageOfocc, yearVeh, airbag, occRole) {
  newdata <- data.frame(
    dvcat=dvcat,  
    seatbelt=seatbelt,
    frontal=frontal,
    sex=sex,
    ageOFocc=as.integer(ageOfocc),
    yearVeh=as.integer(yearVeh),
    airbag=airbag,
    occRole=occRole
    )
  as.numeric(predict(accident_model, newdata, type="response") * 100)
}