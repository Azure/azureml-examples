echo "Creating computes..."

bash -x create-compute.sh

bash -x create-compute-extras.sh



echo "Copying data..."

bash -x copy-data.sh



echo "Creating datasets..."

bash -x create-datasets.sh



echo "Update datasets..."

# bash -x update-datasets.sh



echo "Creating components..."

bash -x create-components.sh




echo "Creating environments..."

bash -x create-environments.sh
