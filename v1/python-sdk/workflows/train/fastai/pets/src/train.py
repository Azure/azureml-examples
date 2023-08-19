import mlflow.fastai
from fastai.vision.all import *

# enable auto logging
# mlflow.fastai.autolog() # broken

path = untar_data(URLs.PETS)
path.ls()

files = get_image_files(path / "images")
len(files)

# (Path('/home/ashwin/.fastai/data/oxford-iiit-pet/images/yorkshire_terrier_102.jpg'),Path('/home/ashwin/.fastai/data/oxford-iiit-pet/images/great_pyrenees_102.jpg'))


def label_func(f):
    return f[0].isupper()


# To get our data ready for a model, we need to put it in a DataLoaders object. Here we have a function that labels using the file names, so we will use ImageDataLoaders.from_name_func. There are other factory methods of ImageDataLoaders that could be more suitable for your problem, so make sure to check them all in vision.data.

dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224))

# We have passed to this function the directory we're working in, the files we grabbed, our label_func and one last piece as item_tfms: this is a Transform applied on all items of our dataset that will resize each imge to 224 by 224, by using a random crop on the largest dimension to make it a square, then resizing to 224 by 224. If we didn't pass this, we would get an error later as it would be impossible to batch the items together.

dls.show_batch()

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)

mlflow.fastai.log_model(learn, "model")
