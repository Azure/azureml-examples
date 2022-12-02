sg_data <- oj_sales |> filter(store == 2, brand == 1)
sg_model <- fit |> filter(store == 2, brand == 1)
sg_fcast <- fcast |> filter(store == 2, brand == 1)
sg_metrics <- metrics |> filter(store == 2, brand == 1)

sg_fcast |> 
  autoplot(sg_data, level = NULL)

# save a model


# https://stackoverflow.com/questions/59721759/extract-model-description-from-a-mable

fit %>% head(2) |> 
  as_tibble() %>%
  gather() %>%
  mutate(model_desc = print(value)) %>%
  select(key, model_desc) %>%
  set_names("model", "model_desc")
