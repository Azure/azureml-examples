# Upload data and prepare jsonl files

Before submitting the job using yaml files, please upload data by running the command below.

```python
python prepare_data.py --subscription <SUBSCRIPTION_ID> --group <RESOURCE_GROUP> --workspace <AML_WORKSPACE_NAME>
```

The script 'prepare_data.py' will
- Download 'fridge items' dataset from a public endpoint
- Upload the data to the workspace provided
- Create jsonl files in data/*-mltable-folder
