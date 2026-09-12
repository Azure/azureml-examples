FROM mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cpu-inference:latest

USER root

RUN NGINX_CONF=/etc/nginx/sites-enabled/app; sed -i "$(grep "location /" $NGINX_CONF -n | cut -f1 -d:) a proxy_buffering off;" $NGINX_CONF;

USER dockeruser

COPY env.yml /tmp/env.yml

RUN conda env update -n amlenv --file /tmp/env.yml