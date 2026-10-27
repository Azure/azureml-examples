# Replace this with the URL of the endpoint
curl "Endpoint URL" \
-H "Content-Type: application/json" \
# Replace this with the primary/secondary key or AMLToken for the endpoint
-H "authorization: AMLToken" \
-d '{
  "prompt": "My name is Julien and I drive to",
  "temperature": 0.8,
  "max_tokens": 100
}'
