FROM rstudio/plumber:latest

ENTRYPOINT []

COPY ./scripts/start_plumber.R /tmp/start_plumber.R 

CMD ["Rscript", "/tmp/start_plumber.R"]