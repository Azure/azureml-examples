import argparse
import os
import shutil
import tempfile
import time

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms as T
from azureml.core import Dataset, Datastore, Run
from azure.storage.blob import BlobServiceClient
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm


LOCAL_RANK = int(os.environ['LOCAL_RANK'])
RANK = int(os.environ['RANK'])
LOCAL_WORLD_SIZE = int(os.environ['AZ_BATCHAI_GPU_COUNT_NEED'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])

LEARNING_RATE = 5e-5
SCALED_LEARNING_RATE = LEARNING_RATE * WORLD_SIZE

OUTPUT_MODELS_DIR = 'model_checkpoints'
MODEL_PATH = f'{OUTPUT_MODELS_DIR}/model.pt'
TRAIN_IMAGE_DOWNLOAD_DIR = os.path.join(tempfile.gettempdir(), 'coco_train_images')
VALID_IMAGE_DOWNLOAD_DIR = os.path.join(tempfile.gettempdir(), 'coco_valid_images')


class MsCocoDataset(torch.utils.data.Dataset):
    """PyTorch dataset for the images in MS COCO."""

    def __init__(self, images_df, images_dir, datastore, download_images):
        self._images_df = images_df
        self._images_dir = images_dir
        self._download_images = download_images
        self._datastore_name = datastore.name

        # Initialize a new blob service client with the latest SDK.
        # Note: using the `azureml._vendor.azure_storage.blob.blockblobservice.BlockBlobService` in the azureml-sdk
        # may cause deadlocks.
        blob_service_client = BlobServiceClient(
            account_url=datastore.blob_service.primary_endpoint,
            credential=datastore.account_key or datastore.sas_token)
        self._container_client = blob_service_client.get_container_client(datastore.container_name)

        self._transform = T.Compose([
            T.ToTensor(),
            # use same transformations as when the model was pre-trained on ImageNet, so transfer learning
            # can apply.
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        image_data = self._images_df.iloc[index]

        # fetch the image from disk or blob storage
        if self._download_images:
            with open(f"{self._images_dir}/{image_data['image_url']}", 'rb') as f:
                image_bytes = f.read()
        else:
            blob_name = str(image_data['image_url'])[len(self._datastore_name)+1:]
            blob_client = self._container_client.get_blob_client(blob_name)
            stream = blob_client.download_blob()
            image_bytes = stream.readall()
        
        # decode the image
        image_bytes = np.asarray(bytearray(image_bytes), dtype="uint8")
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = 1 if image_data['contains_person'] else 0
        label_tensor = torch.as_tensor(label, dtype=torch.float)

        transformed_image = self._transform(image)

        return transformed_image, label_tensor

    def __len__(self):
        return len(self._images_df)


def _get_device():
    if torch.cuda.is_available():
        return torch.device(LOCAL_RANK)
    return torch.device('cpu')

device = _get_device()


def init_optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=SCALED_LEARNING_RATE, momentum=0.9, nesterov=True, weight_decay=1e-4)


def init_model():
    model = torchvision.models.resnet152(pretrained=True)
    num_feats = model.fc.in_features
    model.fc = nn.Linear(num_feats, 1)
    model = model.to(device)
    model = DDP(model)
    return model


def train_epoch(model, train_data_loader, optimizer):
    criterion = nn.BCEWithLogitsLoss()
    for images, targets in tqdm(train_data_loader):

        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs.squeeze(), targets.squeeze())

        start = time.time()
        loss.backward()
        optimizer.step()


def save_checkpoint(model, optimizer, epoch):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, MODEL_PATH)
    shutil.copy(MODEL_PATH, 'outputs/model.pt')


def score_validation_set(model, data_loader):
    print('\nEvaluating validation set accuracy...\n')

    with torch.no_grad():

        num_correct = 0
        num_total_images = 0

        for images, targets in tqdm(data_loader):

            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)

            correct = (outputs.squeeze() > 0.5) == (targets.squeeze() > 0.5)
            num_correct += torch.sum(correct).item()
            num_total_images += len(images)
        
        return num_correct, num_total_images


def load_checkpoint():
    checkpoint = torch.load(MODEL_PATH)
    model = init_model()
    optimizer = init_optimizer(model)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch']


def str2bool(string):
    return string == 'True'


if __name__ == "__main__":

    torch.manual_seed(0)

    # parse input command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_images", type=str2bool, help="Whether to download the images in the dataset to \
        disk, instead of streaming them from storage")
    parser.add_argument("--num_epochs", type=int, help="# of epochs to train the model")
    args, _ = parser.parse_known_args()

    # print parsed arguments
    print('Printing script input arguments...')
    print(f'Download images: {args.download_images}')
    print(f'Number of epochs to train: {args.num_epochs}\n')

    # Turn off parallelism in the Open CV library. Without this line, when multiple training processes are
    # kicked off on the same node, CPU utilization can spike too high and adversely affect training time.
    cv2.setNumThreads(0)

    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    run = Run.get_context()
    workspace = run.experiment.workspace

    dist.init_process_group("nccl", rank=RANK, world_size=WORLD_SIZE)

    datastore = Datastore.get_default(workspace)

    dataset = run.input_datasets['coco_train']
    valid_dataset = run.input_datasets['coco_valid']
    output_dataset = run.output_datasets['model_checkpoints']

    images_df = dataset.to_pandas_dataframe()
    valid_images_df = valid_dataset.to_pandas_dataframe()
    
    if args.download_images:
        print('Downloading images...')
        start = time.time()
        dataset.download('image_url', TRAIN_IMAGE_DOWNLOAD_DIR, overwrite=True)
        valid_dataset.download('image_url', VALID_IMAGE_DOWNLOAD_DIR, overwrite=True)
        image_download_time = time.time() - start
        print(f'Downloaded images in {image_download_time}')

    checkpoint_file_exists = os.path.exists(MODEL_PATH)
    print(f'Checkpoint file exists: {checkpoint_file_exists}')

    # Load state from saved checkpoints if saved checkpoints exist.
    # Otherwise, initialize a model from scratch.
    if checkpoint_file_exists:
        print("Loading saved checkpoints...")
        model, optimizer, starting_epoch = load_checkpoint()
        starting_epoch += 1
    else:
        model = init_model()
        optimizer = init_optimizer(model)
        starting_epoch = 0

    train_dataset = MsCocoDataset(images_df, TRAIN_IMAGE_DOWNLOAD_DIR, datastore, args.download_images)
    valid_dataset = MsCocoDataset(valid_images_df, VALID_IMAGE_DOWNLOAD_DIR, datastore, args.download_images)

    num_workers = os.cpu_count() // LOCAL_WORLD_SIZE
    print(f'Number of PyTorch data loader workers: {num_workers}')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=WORLD_SIZE, rank=RANK)
    train_data_loader = DataLoader(train_dataset, batch_size=32, num_workers=num_workers, sampler=train_sampler, pin_memory=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=32, num_workers=num_workers, pin_memory=True)

    for epoch in range(starting_epoch, args.num_epochs):

        if RANK == 0:
            run.log('training_epoch', epoch)

        print(f'Starting epoch {epoch}')

        model.train()
        train_sampler.set_epoch(epoch)

        start = time.time()
        train_epoch(model, train_data_loader, optimizer)
        if RANK == 0:
            run.log('epoch_train_time', time.time() - start)
        
        if RANK == 0:
            run.flush()
            save_checkpoint(model, optimizer, epoch)    

        model.eval()
        num_correct, num_total_images = score_validation_set(model, valid_data_loader)

        print(f'Scored validation set: {num_correct} correct, {num_total_images} total images')
        validation_accuracy = num_correct / num_total_images * 100
        if RANK == 0:
            run.log('validation_accuracy', validation_accuracy)
        print(f'Accuracy: {validation_accuracy}%')

    print("Done")
