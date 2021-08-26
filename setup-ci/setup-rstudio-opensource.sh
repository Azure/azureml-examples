#!/bin/bash
#This script installs latest version of RStudio open source on compute instance

dpkg --purge rstudio-server # in case open source version is installed.

echo "
[Unit]
Description=RStudio Server
Requires=docker.service
After=docker.service

[Service]
Restart=always
ExecStartPre=-/usr/bin/docker stop rstudio-server
ExecStartPre=-/usr/bin/docker rm rstudio-server
ExecStart=/usr/bin/docker run --privileged --rm -p 8787:8787 -e DISABLE_AUTH=true -v /home/azureuser/cloudfiles/code:/home/azureuser/code -e USER=azureuser --name rstudio-server rocker/rstudio:latest 
ExecStop=/usr/bin/docker stop rstudio-server
[Install]
WantedBy=multi-user.target" | sudo tee /etc/systemd/system/rstudio-server.service

sudo systemctl start rstudio-server
