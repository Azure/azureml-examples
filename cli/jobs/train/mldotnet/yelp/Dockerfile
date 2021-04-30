FROM mcr.microsoft.com/dotnet/sdk:3.1
RUN dotnet tool install -g mlnet

# Install python - temporarily still required to be in your image
RUN add-apt-repository ppa:deadsnakes/ppa \
  apt-get update -qq && \
  apt-get install -y python3.8
 
ENV PATH="$PATH:/root/.dotnet/tools"
