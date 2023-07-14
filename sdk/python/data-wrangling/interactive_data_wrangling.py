## Interactive Data Wrangling using Apache Spark in Azure Machine Learning. Before executing these sample codes in an Azure Machine Learning Notebook, select **Serverless Spark Compute** under **Azure Machine Learning Serverless Spark** or select an attached Synapse Spark pool under **Synapse Spark pools** from the **Compute** selection menu. It is highly recommened to follow the documentation page: [Interactive data wrangling with Apache Spark in Azure Machine Learning](https://learn.microsoft.com/azure/machine-learning/interactive-data-wrangling-with-apache-spark-azure-ml) for more details related to the code samples provided in this notebook.

### Access and wrangle Azure Blob storage data using Access Key

#### First, Set the access key as configuration property `fs.azure.account.key.<STORAGE_ACCOUNT_NAME>.blob.core.windows.net`.

from pyspark.sql import SparkSession

key_vault_name = "<KEY_VAULT_NAME>"
access_key_secret_name = "<ACCESS_KEY_SECRET_NAME>"
storage_account_name = "<STORAGE_ACCOUNT_NAME>"

sc = SparkSession.builder.getOrCreate()
token_library = sc._jvm.com.microsoft.azure.synapse.tokenlibrary.TokenLibrary
access_key = token_library.getSecret(key_vault_name, access_key_secret_name)
sc._jsc.hadoopConfiguration().set(
    f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net", access_key
)

#### Access data using `wasbs://` URI and perform data wrangling.
import pyspark.pandas as pd
from pyspark.ml.feature import Imputer

blob_container_name = "<BLOB_CONTAINER_NAME>"
storage_account_name = "<STORAGE_ACCOUNT_NAME>"

df = pd.read_csv(
    f"wasbs://{blob_container_name}@{storage_account_name}.blob.core.windows.net/data/titanic.csv",
    index_col="PassengerId",
)
imputer = Imputer(inputCols=["Age"], outputCol="Age").setStrategy(
    "mean"
)  # Replace missing values in Age column with the mean value
df.fillna(
    value={"Cabin": "None"}, inplace=True
)  # Fill Cabin column with value "None" if missing
df.dropna(inplace=True)  # Drop the rows which still have any missing value
df.to_csv(
    f"wasbs://{blob_container_name}@{storage_account_name}.blob.core.windows.net/data/wrangled",
    index_col="PassengerId",
)

### Access and wrangle Azure Blob storage data using SAS token

#### First, set the SAS token as configuration property `fs.azure.sas.<BLOB_CONTAINER_NAME>.<STORAGE_ACCOUNT_NAME>.blob.core.windows.net`.
from pyspark.sql import SparkSession

key_vault_name = "<KEY_VAULT_NAME>"
sas_token_secret_name = "<SAS_TOKEN_SECRET_NAME>"
blob_container_name = "<BLOB_CONTAINER_NAME>"
storage_account_name = "<STORAGE_ACCOUNT_NAME>"

sc = SparkSession.builder.getOrCreate()
token_library = sc._jvm.com.microsoft.azure.synapse.tokenlibrary.TokenLibrary
sas_token = token_library.getSecret(key_vault_name, sas_token_secret_name)
sc._jsc.hadoopConfiguration().set(
    f"fs.azure.sas.{blob_container_name}.{storage_account_name}.blob.core.windows.net",
    sas_token,
)

#### Access data using `wasbs://` URI and perform data wrangling.
import pyspark.pandas as pd
from pyspark.ml.feature import Imputer

blob_container_name = "<BLOB_CONTAINER_NAME>"
storage_account_name = "<STORAGE_ACCOUNT_NAME>"

df = pd.read_csv(
    f"wasbs://{blob_container_name}@{storage_account_name}.blob.core.windows.net/data/titanic.csv",
    index_col="PassengerId",
)
imputer = Imputer(inputCols=["Age"], outputCol="Age").setStrategy(
    "mean"
)  # Replace missing values in Age column with the mean value
df.fillna(
    value={"Cabin": "None"}, inplace=True
)  # Fill Cabin column with value "None" if missing
df.dropna(inplace=True)  # Drop the rows which still have any missing value
df.to_csv(
    f"wasbs://{blob_container_name}@{storage_account_name}.blob.core.windows.net/data/wrangled",
    index_col="PassengerId",
)

### Access and wrangle ADLS Gen 2 data using User Identity passthrough

#### - To enable read and write access, assign **Contributor** and **Storage Blob Data Contributor** roles to the user identity.
#### - Access data using `abfss://` URI and perform data wrangling.
import pyspark.pandas as pd
from pyspark.ml.feature import Imputer

file_system_name = "<FILE_SYSTEM_NAME>"
gen2_storage_account_name = "<GEN2_STORAGE_ACCOUNT_NAME>"

df = pd.read_csv(
    f"abfss://{file_system_name}@{gen2_storage_account_name}.dfs.core.windows.net/data/titanic.csv",
    index_col="PassengerId",
)
imputer = Imputer(inputCols=["Age"], outputCol="Age").setStrategy(
    "mean"
)  # Replace missing values in Age column with the mean value
df.fillna(
    value={"Cabin": "None"}, inplace=True
)  # Fill Cabin column with value "None" if missing
df.dropna(inplace=True)  # Drop the rows which still have any missing value
df.to_csv(
    f"abfss://{file_system_name}@{gen2_storage_account_name}.dfs.core.windows.net/data/wrangled",
    index_col="PassengerId",
)


### Access and wrangle data using credentialed AzureML Blob Datastore
#### - Access data using `azureml://` URI and perform data wrangling.
import pyspark.pandas as pd
from pyspark.ml.feature import Imputer

df = pd.read_csv(
    "azureml://datastores/workspaceblobstore/paths/data/titanic.csv",
    index_col="PassengerId",
)
imputer = Imputer(inputCols=["Age"], outputCol="Age").setStrategy(
    "mean"
)  # Replace missing values in Age column with the mean value
df.fillna(
    value={"Cabin": "None"}, inplace=True
)  # Fill Cabin column with value "None" if missing
df.dropna(inplace=True)  # Drop the rows which still have any missing value
df.to_csv(
    "azureml://datastores/workspaceblobstore/paths/data/wrangled",
    index_col="PassengerId",
)

### Access and wrangle data using credentialless AzureML Blob Datastore
#### - To enable read and write access, assign **Contributor** and **Storage Blob Data Contributor** roles to the user identity on the Azure Blob storage account that the datastore points to.
#### - Access data using `azureml://` URI and perform data wrangling.
import pyspark.pandas as pd
from pyspark.ml.feature import Imputer

df = pd.read_csv(
    "azureml://datastores/credlessblobdatastore/paths/data/titanic.csv",
    index_col="PassengerId",
)
imputer = Imputer(inputCols=["Age"], outputCol="Age").setStrategy(
    "mean"
)  # Replace missing values in Age column with the mean value
df.fillna(
    value={"Cabin": "None"}, inplace=True
)  # Fill Cabin column with value "None" if missing
df.dropna(inplace=True)  # Drop the rows which still have any missing value
df.to_csv(
    "azureml://datastores/credlessblobdatastore/paths/data/wrangled",
    index_col="PassengerId",
)
