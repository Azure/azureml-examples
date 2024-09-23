from sys import exit
import json
import requests

class AOAIConstants:
    API_KEY = "<API KEY>"
    ENDPOINT_URL = "<ENDPOINT URL>"
    REQUEST_BODY_TEMPLATE = {
        "messages": [
        {
        "role": "system",
        "content": "How can I help you?"
        }
        ]
    }



class EchoBot:
    async def on_turn(self, context):
        # Check to see if this activity is an incoming message.
        # (It could theoretically be another type of activity.)
        if context.activity.type == "message" and context.activity.text:
            # Check to see if the user sent a simple "quit" message.
            if context.activity.text.lower() == "quit":
                # Send a reply.
                await context.send_activity("Bye!")
                exit(0)
            else:
                user_message = {
                    "role": "user",
                    "content" : context.activity.text
                }
                request_body = AOAIConstants.REQUEST_BODY_TEMPLATE
                request_body['messages'].append(user_message)

                headers = {
                    "api-key": AOAIConstants.API_KEY
                }
                response = requests.post(AOAIConstants.ENDPOINT_URL, headers=headers, json=request_body)
                if response.status_code != 200:
                    await context.send_activity(f"An error occurred in querying the AOAI endpoint: {response.text}")

                response_json = json.loads(response.text)
                chatbot_message = response_json['choices'][0]['message']['content']
                await context.send_activity(f"Assistant: {chatbot_message}\n Is there anything else I can help you with?")
