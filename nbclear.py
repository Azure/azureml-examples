import os 
import glob

nbs = glob.glob('**/*.ipynb', recursive=True)

for nb in nbs:
    os.system(f'jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {nb}')