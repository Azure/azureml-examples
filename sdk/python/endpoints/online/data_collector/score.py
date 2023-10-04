import pandas as pd
import json
from azureml.ai.monitoring import Collector


def init():
    global inputs_collector, outputs_collector, inputs_outputs_collector

    # instantiate collectors with appropriate names, make sure align with deployment spec
    inputs_collector = Collector(name="model_inputs")
    outputs_collector = Collector(name="model_outputs")
    inputs_outputs_collector = Collector(
        name="model_inputs_outputs"
    )  # note: this is used to enable Feature Attribution Drift


def run(data):
    # json data: { "data" : {  "col1": [1,2,3], "col2": [2,3,4] } }
    pdf_data = preprocess(json.loads(data))

    # tabular data: {  "col1": [1,2,3], "col2": [2,3,4] }
    input_df = pd.DataFrame(pdf_data)

    # collect inputs data, store correlation_context
    context = inputs_collector.collect(input_df)

    # perform scoring with pandas Dataframe, return value is also pandas Dataframe
    output_df = predict(input_df)

    # collect outputs data, pass in correlation_context so inputs and outputs data can be correlated later
    outputs_collector.collect(output_df, context)

    # create a dataframe with inputs/outputs joined - this creates a URI folder (not mltable)
    # input_output_df = input_df.merge(output_df, context)
    input_output_df = input_df.join(output_df)

    # collect both your inputs and output
    inputs_outputs_collector.collect(input_output_df, context)

    return output_df.to_dict()


def preprocess(json_data):
    # preprocess the payload to ensure it can be converted to pandas DataFrame
    return json_data["data"]


def predict(input_df):
    # process input and return with outputs
    ...

    return output_df
