#!/bin/sh

echo "Add R-base Repository....."
add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran40/'
echo "Running System update....."
apt-get update -y
echo "Install R programming language....."
apt-get install r-base r-base-core r-recommended -y
echo "Finished R programming language Installation...."
wget --quiet -O "rstudio-server-1.4.1717-amd64.deb" https://download2.rstudio.org/server/bionic/amd64/rstudio-server-1.4.1717-amd64.deb
echo "y" | sudo gdebi rstudio-server-1.4.1717-amd64.deb
echo "Finished installing RStudio Server....."
echo "Install Rstudio Desktop for Rstudio"
echo "downloading rstudio desktop...."
wget https://download1.rstudio.org/desktop/bionic/amd64/rstudio-1.4.1717-amd64.deb
apt install ./rstudio-1.4.1717-amd64.deb -y
echo "Finished installing RStudio Desktop....."
