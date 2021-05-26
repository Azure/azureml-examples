# Creating a job through REST

1. Upload the files in the [model-1](/model-1) to the storage container associated with your workspace
2. Run the REST requests with the payloads in this folder in the following order:
- Create datastore
- Create code version
- Create model version
- Create environment
- Create endpoint
- Create deployment

The request URIs and required headers can be found in the postman collection uploaded to the [rest-api folder](../../..). This postman collection contains all of the requests possible with this API version.