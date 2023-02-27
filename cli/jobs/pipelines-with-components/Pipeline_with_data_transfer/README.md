## Move data between different source and use this job as one step of pipeline job
We support to move data assets such as folder to different different data store. Meanwhile, we also support customer to move external data such as s3 or sql based data such as azure SQL or snowflakes to AzureML as mltable.
## task_type
We support following task_types:
- import_data:
    - import external folder data such as s3 to AzureML as uri_folder data asset, import sql based data such as azure SQL or snowflakes to AzureML as mltable data asset
    - We will provide build-in component in system registry, customer can directly use it in pipeline designer canvas.
- export_data: export mltable data asset to internal or external sql source such as azure SQL or snowflakes.
- copy_data: you can use copy_data to copy data assets to different data storage or other azure cloud storage.
## compute experience
We introduce serverless compute experience for data transfer job. To use serverless compute in pipeline, you can specify `default_compute` under `settings` as `serverless` or you can specify compute in data transfer step as `serverless`.