import json

with open('tutorials/azureml-in-a-day/azureml-in-a-day.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any('blue_deployment = ManagedOnlineDeployment' in s for s in cell['source']):
        # Insert environment line after model=model line (index 8)
        cell['source'].insert(9, '    environment=f"{custom_env_name}@latest",\n')
        print('Updated cell:')
        for i, line in enumerate(cell['source']):
            print(f'  {i}: {repr(line)}')
        break

with open('tutorials/azureml-in-a-day/azureml-in-a-day.ipynb', 'w', newline='\n') as f:
    json.dump(nb, f, indent=1)
print('Done')
