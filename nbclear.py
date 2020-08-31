import os 
import glob

nbs = glob.glob('notebooks/**/*.ipynb', recursive=True)

for nb in nbs:
    os.system(f'jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {nb}')

nbs = glob.glob('concepts/**/*.ipynb', recursive=True)

for nb in nbs:
    os.system(f'jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {nb}')