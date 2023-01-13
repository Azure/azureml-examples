FROM rstudio/plumber:latest

# Install carrier package to be able to use the crated model you created in 
# the model training job
RUN R -e "install.packages('carrier', dependencies = TRUE, repos = 'https://cloud.r-project.org/')"

# Install any additional R packages you may need for your model crate to run

RUN R -e "install.packages('tsibble', dependencies = TRUE, repos = 'https://cloud.r-project.org/')"
RUN R -e "install.packages('fable', dependencies = TRUE, repos = 'https://cloud.r-project.org/')"

ENTRYPOINT []

COPY ./start_plumber.R /tmp/start_plumber.R 

CMD ["Rscript", "/tmp/start_plumber.R"]