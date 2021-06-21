# plumber.R

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

#* Return the sum of two numbers
#* @param a The first number to add
#* @param b The second number to add
#* @post /score
function(a, b) {
  as.numeric(a) + as.numeric(b)
}