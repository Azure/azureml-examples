wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
echo "unzipping to $1"
tar -xvzf cifar-10-python.tar.gz -C $1
rm cifar-10-python.tar.gz
ls $1

