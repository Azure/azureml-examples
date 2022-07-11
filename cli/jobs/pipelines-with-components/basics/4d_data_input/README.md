This example show how to use legacy v1 dataset and data asset in v2. 
- When using v1 file dataset in v2, you need use `type`: mltable and `mode`: eval_mount or eval_download.
- To create a legacy v1 dataset, you can using UI or legacy SDK v1. Data asset create using CLI v2 will create a v2 data asset.
- To create a v2 data asset please using `az ml data create -f data.yml`