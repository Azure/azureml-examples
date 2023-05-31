# Running a Pipeline job using data transfer
In this example, we will explains how to create a data transfer node and use it in a pipeline. A data transfer component can be used to copy data in different data storage. It support following scenarios:
- import: import external file_system such as s3, and SQL database such as snowflake, azure sql, as azureml dataaset. 
- export: export azureml dataset to external file_system or SQL database.
- copy: copy data between azureml data assets.
We will use connection to authenticate to external data source and SQL database. For more examples about connection, please refer to [this document](https://github.com/Azure/azureml-examples/tree/main/cli/resources/connections).

```yaml
# file system can use as source and sink in top level property
source:
    type: file_system
    path: ./path/on/s3/to/data/
    connection: azureml:my_s3_connection    

sink:
    type: file_system
    path: ./path/on/s3/to/data/
    connection: azureml:my_s3_connection 

# database use as source in top level property - query
source:
    type: database
    query: >-
      SELECT * 
      FROM my_table
    connection: azureml:my_sql_connection  

# database use as source in top level property - store procedure
source:
    type: database
    stored_procedure: SelectEmployeeByJobAndDepartment
    stored_procedure_params:
      - name: job
        value: Engineer
        type:  String
      - name: department
        value: Engineering
        type:  String
    connection: azureml:my_sql_connection  

## stored_procedure_params is optional, customer can use stored_procedure without params

# database use as sink in top level property
sink:
    type: database
    table_name: my_table # table name
    connection: azureml:my_sql_connection
```