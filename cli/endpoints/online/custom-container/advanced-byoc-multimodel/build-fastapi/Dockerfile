FROM python:3.9-slim-bullseye

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \ 
    APP_PARENT=/var/abyoc \
    APP_ROOT=/var/abyoc/app \
    MODEL_DIR=/var/abyoc/models \
    APP_VENV=/opt/envs/abyoc \ 
    WORKER_COUNT=2

COPY . $APP_PARENT

RUN apt-get update && \
    apt-get install -y --no-install-recommends nginx-light sudo && \ 
    apt-get autoremove -y && \
    apt-get clean -y &&  \
    rm -rf /usr/share/man/* && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p $APP_VENV && python3 -m venv $APP_VENV && \
    mkdir -p $MODEL_DIR $APP_ROOT && \ 
    mv $APP_PARENT/uvicorn_app /etc/nginx/sites-available/ && \
    rm /etc/nginx/sites-enabled/default && \ 
    ln -s /etc/nginx/sites-available/uvicorn_app /etc/nginx/sites-enabled/ && \
    useradd --create-home dockeruser && \ 
    usermod -aG sudo dockeruser && \ 
    echo "dockeruser ALL=(ALL:ALL) NOPASSWD:/usr/sbin/service nginx start" >> /etc/sudoers.d/dockeruser

ENV PATH=$APP_VENV/bin:$PATH \
    LD_LIBRARY_PATH=$APP_VENV/lib:$PATH

RUN pip install --no-cache-dir --upgrade -r $APP_PARENT/requirements.txt

WORKDIR $APP_ROOT

USER dockeruser

CMD sudo service nginx start && uvicorn fastapi-main:app --uds /tmp/uvicorn.sock --log-level trace --timeout-keep-alive 60 --workers 4