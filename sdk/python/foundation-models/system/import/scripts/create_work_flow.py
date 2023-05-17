from huggingface_hub import HfApi, ModelFilter

def get_top_model_ids(task, sort_key = "downloads", direction = -1, limit = 10):
    api = HfApi()
    models = api.list_models(
                         filter=ModelFilter(
                             task=task,
                         ),
                         sort = sort_key,
                         direction = direction,
                         limit = limit
                         )
    models = list(models)
    return [x.modelId for x in models]

def generate_workflow_file(template, parameters, output_path):
    with open(template, 'r') as f:
        template_content = f.read()

    for parameter in parameters:
        # Replace placeholders in the template with the parameter values
        workflow_content = template_content.replace('<model-id>', parameter)
        # Create a new workflow file with the parameter-specific content
        output_file = f"{output_path}/import-{parameter.replace('/','-')}.yaml"
        with open(output_file, 'w') as f:
            f.write(workflow_content)

        print(f"Created workflow file: {output_file}")

def create_md_table(data):
    table_header = "| Task | Model ID | Status |\n"
    table_divider = "| --- | --- | --- |\n"

    table_rows = ""
    for row in data:
        task_supported = row['task_supported']
        model_id = row['model_id']
        table_rows += f"| {task_supported} | {model_id} | [![Importing {model_id}](https://github.com/Azure/azureml-examples/actions/workflows/import-{model_id.replace('/','-')}.yaml/badge.svg?branch=hrishikesh/model-import-workflows)](https://github.com/Azure/azureml-examples/actions/workflows/import-{model_id.replace('/','-')}.yaml?branch=hrishikesh/model-import-workflows) |\n"

    table = table_header + table_divider + table_rows

    with open('../README.md', 'w') as file:
        file.write(table)
    print("README file created which will show the status of import workflow.....")
# Usage example
template_file = 'workflow_template.yaml'  # Path to your template workflow file
output_directory = '../../../../../../.github/workflows'  # Directory where the generated workflow files will be saved

task_supported = [
'fill-mask',
'token-classification',
'question-answering',
'summarization',
'text-generation',
'text-classification',
'translation',
'image-classification',
'text-to-image'
]
data = []
for task in task_supported:
    model_ids = get_top_model_ids(task = task)
    generate_workflow_file(template_file, model_ids, output_directory)
    for model_id in model_ids:
        data.append({'model_id':model_id, 'task_supported':task})
    print(f"Workflow file generated for task {task}")

create_md_table(data)

#### To do
## Remove workflow files if not in top n models
## Add better function to create md file