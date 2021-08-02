# imports
import os
import json
import yaml
import argparse

from adlfs import AzureBlobFileSystem as abfs

# define functions
def main(args):
    # build storage options dictionary
    storage_options = {"account_name": args.account_name}
    schemas = get_schemas(storage_options, args.container)

    # make directories
    os.makedirs(f"jsons/{args.container}", exist_ok=True)
    os.makedirs(f"yamls/{args.container}", exist_ok=True)

    # process each schema
    for schema in schemas:
        write_json(schema, schemas[schema])
        write_yaml(schema, schemas[schema])


def write_json(filename, schema):
    # write schema to json
    with open(f"jsons/{filename}", "w") as f:
        json.dump(schema, f, indent=2)


def write_yaml(filename, schema):
    # write schema to yaml
    with open(f"yamls/{filename}".replace("json", "yml"), "w") as f:
        yaml.dump(schema, f, indent=2)


def get_schemas(storage_options, container):
    # initialize azure blob file system
    fs = abfs(**storage_options)

    # get list of files under container
    files = fs.ls(container)

    # filter to json schemas
    schemas = [f for f in files if "schema.json" in f]

    # read in schemas
    schemas = {schema: json.load(fs.open(schema)) for schema in schemas}

    # return schemas
    return schemas


# run functions
if __name__ == "__main__":
    # setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schema-url", type=str, default="https://azuremlschemas.azureedge.net"
    )
    parser.add_argument("--account-name", type=str, default="azuremlsdk2")
    parser.add_argument("--container", type=str, default="latest")
    args = parser.parse_args()

    # call main
    main(args)
