#!/usr/bin/env python3

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from azure.identity import DeviceCodeCredential


# Initialize AAD (prompts the user with an AAD device code)
credential = DeviceCodeCredential()
credential.get_token("https://management.azure.com/.default")


class MockMsiServer(BaseHTTPRequestHandler):
    def do_GET(self):
        access_token = credential.get_token(
            "https://management.azure.com/.default"
        ).token
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(bytes(json.dumps({"access_token": access_token}), "utf8"))


with HTTPServer(("127.0.0.1", 46808), MockMsiServer) as server:
    print(f"Starting the mock MSI server at {server.server_address}")
    server.serve_forever()
