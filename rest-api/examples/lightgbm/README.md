# Creating a job through REST

1. Upload the files in the [iris folder](/iris) to the storage container associated with your workspace
2. Run the REST requests with the payloads in this folder in the following order:
- Create datastore
- Create code version
- Create data version
- Create environment
- Create command job/create sweep job

The request URIs and headers can be found in the postman collection uploaded to the [rest-api folder](../../..). This postman collection contains all of the requests possible with this API version.