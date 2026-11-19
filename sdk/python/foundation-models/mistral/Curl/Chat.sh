# Replace this with the URL of the endpoint
curl "Endpoint URL" \
-H "Content-Type: application/json" \
# Replace this with the primary/secondary key or AMLToken for the endpoint
-H "authorization: AMLToken" \
-d '{
  "messages": [
    {
      "role": "user",
      "content": "What is the second largest ocean in the world?"
    },
    {
      "role": "assistant",
      "content": "The largest ocean in the world is the Pacific Ocean, which covers an area of approximately one fifty five million square kilometers sixty million square miles."
    }
  ],
  "temperature": 0.8,
  "max_tokens": 128
}'
