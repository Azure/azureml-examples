import shutil
import os
import yaml

def create_ml_table(csv_file, output, delimiter=',', encoding='ascii'):
    os.makedirs(output, exist_ok=True)
    fname = os.path.split(csv_file)[-1]
    mltable = {
        'paths': [{'file': f'./{fname}'}],
        'transformations': [
                {'read_delimited': {
                    'delimiter': delimiter,
                    'encoding': encoding
                }}
            ]
    }
    with open(os.path.join(output, 'MLTable'), 'w') as f:
        f.write(yaml.dump(mltable))
    shutil.copy(csv_file, os.path.join(output, fname))