from azureml.core import Run, Dataset
import os


import glob
import os
import sys


from os import listdir
from os.path import isfile, join
print(sys.argv[0])
print(sys.argv[1])
for item in os.environ:
    print(item)
mypath = os.environ["input2"]
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for file in files:
    with open(join(mypath,file)) as f:
        print(f.read())