fit |> 
  select(arima) |> 
  tidy()


sg_model

sg_model |> 
  select(mean) |> 
  report()

sg_model |> 
  select(naive) |> 
  report()

sg_model |> 
  select(drift) |> 
  report()

sg_model |> 
  select(arima) |> 
  tidy()

sg_model |> 
  pull(arima) |> 
  tidy()

sg_model |> class()

a <- sg_model |> 
  select(arima) |> 
  pluck() 
b <- sg_model |> 
  pull(arima)




mdl <- sg_model |> 
  pull(arima) 

forecast(mdl, h = 1000)



