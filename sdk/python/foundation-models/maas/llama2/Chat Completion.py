# Import necessary Library
import urllib.request
import json
import os
import ssl
from dotenv import load_dotenv

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

# Request data goes here
# The example below assumes JSON formatting which may be updated
# depending on the format your endpoint expects.
# More information can be found here:
# https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
data =  {
  "prompt": "My name is Julien and I like to",
  "temperature": 0.8,
  "max_tokens": 128
}

body = str.encode(json.dumps(data))

#Load the environmet variable from the .env file
load_dotenv()
# Load the URL from the .env file using this os.getenv()
url = os.getenv("CHAT_COMPLETION_API_URL")
# Load the primary/secondary key or AMLToken for the endpoint from the .env file using os.getenv()
api_key = os.getenv("CHAT_COMPLETION_API_KEY")

if not api_key:
    raise Exception("A key should be provided to invoke the endpoint")


headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(error.read().decode("utf8", 'ignore'))