if [ $RANK -eq 0 ]
then 
    cp -r mlruns/*/*/artifacts/model $1
fi