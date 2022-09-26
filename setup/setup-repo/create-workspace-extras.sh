locations=("eastus" "westus" "westus2" "southcentralus")

for location in "${locations[@]}"

do

  az ml workspace create -n "main-$location" -l $location --no-wait

done

