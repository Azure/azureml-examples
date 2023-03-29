# import library to parse command line arguments
import argparse, os
parser = argparse.ArgumentParser()
# add an argument to specify a dataset name to download
parser.add_argument('--dataset', type=str, default='conll2003', help='dataset name')
# add an argument to specify the directory to download the dataset to
parser.add_argument('--download_dir', type=str, default='data', help='directory to download the dataset to')
args = parser.parse_args()

# create the download directory if it does not exist
if not os.path.exists(args.download_dir):
    os.makedirs(args.download_dir)

def format_ner_tags(example, class_names):
    example['text'] = ' '.join(example['tokens'])
    example['ner_tags_str'] = [class_names[id] for id in example['ner_tags']]
    return example

# import hugging face datasets library
from datasets import load_dataset, get_dataset_split_names
from functools import partial
for split in get_dataset_split_names(args.dataset):
    # load the split of the dataset
    dataset = load_dataset(args.dataset, split=split)
    dataset = dataset.map(
        partial(
            format_ner_tags,
            class_names=dataset.features['ner_tags'].feature.names
        )
    )
    # save the split of the dataset to the download directory as json lines file
    dataset.to_json(os.path.join(args.download_dir, f'{split}.jsonl'))
    # print dataset features

