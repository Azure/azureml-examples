from azureml.core import Run

input_file_ds_path = Run.get_context().input_datasets["file_dataset"]
with open(input_file_ds_path, "r") as f:
    content = f.read()
    print(content)

input_tabular_ds = Run.get_context().input_datasets["tabular_dataset"]
tabular_df = input_tabular_ds.to_pandas_dataframe()
print(tabular_df)
