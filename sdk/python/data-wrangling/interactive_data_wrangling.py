## Interactive Data Wrangling using Apache Spark in Azure Machine Learning. Before executing these sample codes in an Azure Machine Learning Notebook, select **Serverless Spark Compute** under **Azure Machine Learning Serverless Spark** or select an attached Synapse Spark pool under **Synapse Spark pools** from the **Compute** selection menu. It is highly recommened to follow the documentation page: [Interactive data wrangling with Apache Spark in Azure Machine Learning](https://learn.microsoft.com/azure/machine-learning/interactive-data-wrangling-with-apache-spark-azure-ml) for more details related to the code samples provided in this notebook.

### Access and wrangle Azure Blob storage data using Access Key

#### First, Set the access key as configuration property `fs.azure.account.key.<STORAGE_ACCOUNT_NAME>.blob.core.windows.net`.

from pyspark.sql import SparkSession

sc = SparkSession.builder.getOrCreate()
token_library = sc._jvm.com.microsoft.azure.synapse.tokenlibrary.TokenLibrary
access_key = token_library.getSecret("<KEY_VAULT_NAME>", "<ACCESS_KEY_SECRET_NAME>")
sc._jsc.hadoopConfiguration().set(
    "fs.azure.account.key.<STORAGE_ACCOUNT_NAME>.blob.core.windows.net", access_key
)

#### Access data using `wasbs://` URI and perform data wrangling.
import pyspark.pandas as pd
from pyspark.ml.feature import Imputer

df = pd.read_csv(
    "wasbs://<BLOB_CONTAINER_NAME>@<STORAGE_ACCOUNT_NAME>.blob.core.windows.net/data/titanic.csv",
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
    "wasbs://<BLOB_CONTAINER_NAME>@<STORAGE_ACCOUNT_NAME>.blob.core.windows.net/data/wrangled",
    index_col="PassengerId",
)

### Access and wrangle Azure Blob storage data using SAS token

#### First, set the SAS token as configuration property `fs.azure.sas.<BLOB_CONTAINER_NAME>.<STORAGE_ACCOUNT_NAME>.blob.core.windows.net`.
from pyspark.sql import SparkSession

sc = SparkSession.builder.getOrCreate()
token_library = sc._jvm.com.microsoft.azure.synapse.tokenlibrary.TokenLibrary
sas_token = token_library.getSecret("<KEY_VAULT_NAME>", "<SAS_TOKEN_SECRET_NAME>")
sc._jsc.hadoopConfiguration().set(
    "fs.azure.sas.<BLOB_CONTAINER_NAME>.<STORAGE_ACCOUNT_NAME>.blob.core.windows.net",
    sas_token,
)

#### Access data using `wasbs://` URI and perform data wrangling.
import pyspark.pandas as pd
from pyspark.ml.feature import Imputer

df = pd.read_csv(
    "wasbs://<BLOB_CONTAINER_NAME>@<STORAGE_ACCOUNT_NAME>.blob.core.windows.net/data/titanic.csv",
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
    "wasbs://<BLOB_CONTAINER_NAME>@<STORAGE_ACCOUNT_NAME>.blob.core.windows.net/data/wrangled",
    index_col="PassengerId",
)

### Access and wrangle ADLS Gen 2 data using User Identity passthrough

#### - To enable read and write access, assign **Contributor** and **Storage Blob Data Contributor** roles to the user identity.
#### - Access data using `abfss://` URI and perform data wrangling.
import pyspark.pandas as pd
from pyspark.ml.feature import Imputer

df = pd.read_csv(
    "abfss://<FILE_SYSTEM_NAME>@<GEN2_STORAGE_ACCOUNT_NAME>.dfs.core.windows.net/data/titanic.csv",
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
    "abfss://<FILE_SYSTEM_NAME>@<GEN2_STORAGE_ACCOUNT_NAME>.dfs.core.windows.net/data/wrangled",
    index_col="PassengerId",
)

### Access and wrangle ADLS Gen 2 data using Service Principal

#### - To enable read and write access, assign **Contributor** and **Storage Blob Data Contributor** roles to the Service Principal.
#### - Set configuration properties as follows:
####     - Client ID property: `fs.azure.account.oauth2.client.id.<STORAGE_ACCOUNT_NAME>.dfs.core.windows.net`
####     - Client secret property: `fs.azure.account.oauth2.client.secret.<STORAGE_ACCOUNT_NAME>.dfs.core.windows.net`
####     - Tenant ID property: `fs.azure.account.oauth2.client.endpoint.<STORAGE_ACCOUNT_NAME>.dfs.core.windows.net`
####     - Tenant ID value: `https://login.microsoftonline.com/<TENANT_ID>/oauth2/token`
from pyspark.sql import SparkSession

sc = SparkSession.builder.getOrCreate()
token_library = sc._jvm.com.microsoft.azure.synapse.tokenlibrary.TokenLibrary

# Set up service principal tenant ID, client ID and secret from Azure Key Vault
client_id = token_library.getSecret("<KEY_VAULT_NAME>", "<CLIENT_ID_SECRET_NAME>")
tenant_id = token_library.getSecret("<KEY_VAULT_NAME>", "<TENANT_ID_SECRET_NAME>")
client_secret = token_library.getSecret("<KEY_VAULT_NAME>", "<CLIENT_SECRET_NAME>")

# Set up service principal which has access of the data
sc._jsc.hadoopConfiguration().set(
    "fs.azure.account.auth.type.<STORAGE_ACCOUNT_NAME>.dfs.core.windows.net", "OAuth"
)
sc._jsc.hadoopConfiguration().set(
    "fs.azure.account.oauth.provider.type.<STORAGE_ACCOUNT_NAME>.dfs.core.windows.net",
    "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
)
sc._jsc.hadoopConfiguration().set(
    "fs.azure.account.oauth2.client.id.<STORAGE_ACCOUNT_NAME>.dfs.core.windows.net",
    client_id,
)
sc._jsc.hadoopConfiguration().set(
    "fs.azure.account.oauth2.client.secret.<STORAGE_ACCOUNT_NAME>.dfs.core.windows.net",
    client_secret,
)
sc._jsc.hadoopConfiguration().set(
    "fs.azure.account.oauth2.client.endpoint.<STORAGE_ACCOUNT_NAME>.dfs.core.windows.net",
    "https://login.microsoftonline.com/" + tenant_id + "/oauth2/token",
)

#### - Access data using `abfss://` URI and perform data wrangling.
import pyspark.pandas as pd
from pyspark.ml.feature import Imputer

df = pd.read_csv(
    "abfss://<FILE_SYSTEM_NAME>@<GEN2_STORAGE_ACCOUNT_NAME>.dfs.core.windows.net/data/titanic.csv",
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
    "abfss://<FILE_SYSTEM_NAME>@<GEN2_STORAGE_ACCOUNT_NAME>.dfs.core.windows.net/data/wrangled",
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

### Access credentialed AzureML ADLS Gen 2 Datastore
#### - To enable read and write access, assign **Contributor** and **Storage Blob Data Contributor** roles to the service principal used by datastore.
#### - Access data using `azureml://` URI and perform data wrangling.
import pyspark.pandas as pd
from pyspark.ml.feature import Imputer

df = pd.read_csv(
    "azureml://datastores/adlsg2datastore/paths/data/titanic.csv",
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
    "azureml://datastores/credadlsg2datastore/paths/data/wrangled",
    index_col="PassengerId",
)

### Access and wrangle data using credentialless AzureML ADLS Gen 2 Datastore
#### - To enable read and write access, assign **Contributor** and **Storage Blob Data Contributor** roles to the user identity on the Azure Data Lake Storage (ADLS) Gen 2 storage account that the datastore points to.
#### - Access data using `azureml://` URI and perform data wrangling.
import pyspark.pandas as pd
from pyspark.ml.feature import Imputer

df = pd.read_csv(
    "azureml://datastores/credlessadlsg2datastore/paths/data/titanic.csv",
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
    "azureml://datastores/credlessadlsg2datastore/paths/data/wrangled",
    index_col="PassengerId",
)

### Access mounted File Share
#### Access data on mounted File share by constructing absolute path.
import os
import pyspark.pandas as pd

abspath = os.path.abspath(".")
file = "file://" + abspath + "/Users/<USER>/data/titanic.csv"
print(file)
df = pd.read_csv(file)
df.head()
