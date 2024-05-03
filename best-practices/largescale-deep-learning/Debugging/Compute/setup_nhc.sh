#!/bin/bash

apt-get update -y
apt-get install python3.10 python3-pip -y
pip install requests
python3 run_nhc.py