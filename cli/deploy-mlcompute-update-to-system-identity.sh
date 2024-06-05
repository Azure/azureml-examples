export COMPUTE_NAME=mycluster-sa

does_compute_exist()
{
  if [ -z $(az ml compute show -n $COMPUTE_NAME --query name) ]; then
    echo false
  else
    echo true
  fi
}

echo "Checking if compute $COMPUTE_NAME already exists"
if [ "$(does_compute_exist)" == "true" ]; then
  echo "Skipping, compute: $COMPUTE_NAME exists"
else
  echo "Provisioning compute: $COMPUTE_NAME"
  az ml compute create --name "$COMPUTE_NAME" --type amlcompute
fi

az ml compute update --name "$COMPUTE_NAME" --identity-type system_assigned