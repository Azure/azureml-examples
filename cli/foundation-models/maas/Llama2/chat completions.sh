# Replace this with the URL of the endpoint
curl "https://Llama-2-13b-tcrwa-serverless.eastus2.inference.ai.azure.com/v1/completions" \
-H "Content-Type: application/json" \
# Replace this with the primary/secondary key or AMLToken for the endpoint
-H "authorization: AMLToken" \
-d '{
  "prompt": "My name is Julien and I drive to",
  "temperature": 0.8,
  "max_tokens": 100
}'