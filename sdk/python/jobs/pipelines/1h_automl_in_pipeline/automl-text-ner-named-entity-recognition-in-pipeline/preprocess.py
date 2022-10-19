import argparse
import datetime
from pathlib import Path
import yaml
from mltable import load

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--train_data", type=str)
    parser.add_argument(
        "--validation_data", type=str
    )
    parser.add_argument("--preprocess_train_data", type=str)
    parser.add_argument(
        "--preprocess_validation_data", type=str
    )
    # parse args
    args = parser.parse_args()
    print("args received ", args)
    # return args
    return args

def get_preprocessed_data(text):
    """
    Do preprocessing as needed
    Currently we are just passing text file as it is
    """
    return text

def main(args):
    """
    Preprocessing of training/validation data
    """
    with open(args.train_data+'/train.txt') as f:
        train_data = f.read()
    preprocessed_train_data = get_preprocessed_data(train_data)   
    
    # write preprocessed train txt file
    with open(args.preprocess_train_data+'/train.txt', 'w') as f:
        f.write(preprocessed_train_data)
    
    
    with open(args.validation_data+'/valid.txt') as f:
        validation_data = f.read()
    preprocessed_validation_data = get_preprocessed_data(validation_data)   
    
    # write preprocessed validation txt file
    with open(args.preprocess_validation_data+'/valid.txt', 'w') as f:
        f.write(preprocessed_validation_data) 
    
    # Write MLTable yaml file as well in output folder
    # Since in this example we are not doing any preprocessing, we are just copying same yaml file from input,change it if needed
   
    # read and write MLModel yaml file for train data
    with open(args.train_data+'/MLTable','r') as file:
        yaml_file = yaml.safe_load(file)
    with open(args.preprocess_train_data+'/MLTable', 'w') as file:
        yaml.dump(yaml_file, file)
    
    
    # read and write MLModel yaml file for validation data
    with open(args.validation_data+'/MLTable','r') as file:
        yaml_file = yaml.safe_load(file)
    with open(args.preprocess_validation_data+'/MLTable', 'w') as file:
        yaml.dump(yaml_file, file)
        
# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)