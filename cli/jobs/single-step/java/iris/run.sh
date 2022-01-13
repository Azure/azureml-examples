#az login 

az configure --defaults group=mlstudio workspace=ml-cust-demo location=EastUS

az ml job create --file job-template.yml --web

