from pathlib import Path
import sys
import runpy
import json
import shutil
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
import functools
from enum import Enum
from azure.ml.component import dsl
from azure.ml.component.dsl.types import Integer, String, Float, Enum as EnumParameter


class Data_BackendEnum(Enum):
    pytorch = "pytorch"
    syntetic = "syntetic"
    dali_gpu = "dali-gpu"
    dali_cpu = "dali-cpu"


class ArchEnum(Enum):
    resnet18 = "resnet18"
    resnet34 = "resnet34"
    resnet50 = "resnet50"
    resnet101 = "resnet101"
    resnet152 = "resnet152"
    resnext101_32x4d = "resnext101-32x4d"
    se_resnext101_32x4d = "se-resnext101-32x4d"


class Model_ConfigEnum(Enum):
    classic = "classic"
    fanin = "fanin"
    grp_fanin = "grp-fanin"
    grp_fanout = "grp-fanout"


class Lr_ScheduleEnum(Enum):
    step = "step"
    linear = "linear"
    cosine = "cosine"


def convert_image_directory_to_specific_format(
    image_dir_path, output_root, is_train=False
):
    # convert image directory to train component input data format
    image_dir_path = Path(image_dir_path)
    image_list_path = image_dir_path / "images.lst"
    output_data_path = output_root / ("train" if is_train else "val")
    category_list = []
    file_name_list = []
    with open(image_list_path, "r") as fin:
        for line in fin:
            line = json.loads(line)
            # print(line)
            category_list.append(line["category"])
            file_name_list.append(line["image_info"]["file_name"])
            (output_data_path / line["category"]).mkdir(parents=True, exist_ok=True)
    print(
        f"file number {len(file_name_list)}, category number {len(set(category_list))}."
    )

    def copy_file(index):
        target_dir = output_data_path / category_list[index]
        shutil.copyfile(
            str(image_dir_path / file_name_list[index]),
            str(target_dir / Path(file_name_list[index]).name),
        )

    with ThreadPool(cpu_count()) as p:
        p.map(functools.partial(copy_file), range(len(file_name_list)))

    print(
        f"output path {output_data_path} has {len(list(output_data_path.glob('**/*')))} files."
    )
    return output_root


@dsl._component()
def main(
    train_data: String(description="path to train dataset") = None,
    val_data: String(description="path to valid dataset") = None,
    data_backend: EnumParameter(
        enum=Data_BackendEnum,
        description="data backend: pytorch | syntetic | dali-gpu | dali-cpu (default: dali-cpu)",
    ) = Data_BackendEnum.dali_cpu,
    arch: EnumParameter(
        enum=ArchEnum,
        description="model architecture: resnet18 | resnet34 | resnet50 | resnet101 | resnet152 | resnext101_32x4d | se_resnext101_32x4d (default: resnet50)",
    ) = ArchEnum.resnet50,
    model_config: EnumParameter(
        enum=Model_ConfigEnum,
        description="model configs: classic | fanin | grp_fanin | grp_fanout(default: classic)",
    ) = Model_ConfigEnum.classic,
    workers: Integer(description="number of data loading workers (default: 5)") = 5,
    epochs: Integer(description="number of total epochs to run") = 90,
    batch_size: Integer(description="mini-batch size (default: 256) per gpu") = 256,
    optimizer_batch_size: Integer(
        description="size of a total batch size, for simulating bigger batches using gradient accumulation"
    ) = -1,
    lr: Float(description="initial learning rate") = 0.1,
    lr_schedule: EnumParameter(
        enum=Lr_ScheduleEnum, description="Type of LR schedule: step, linear, cosine"
    ) = Lr_ScheduleEnum.step,
    warmup: Integer(description="number of warmup epochs") = 0,
    label_smoothing: Float(description="label smoothing") = 0.0,
    mixup: Float(description="mixup alpha") = 0.0,
    momentum: Float(description="momentum") = 0.9,
    weight_decay: Float(description="weight decay (default: 1e-4)") = 0.0001,
    print_freq: Integer(description="print frequency (default: 10)") = 10,
    resume: String(description="path to latest checkpoint (default: none)") = "",
    pretrained_weights: String(description="load weights from here") = "",
    static_loss_scale: Float(
        description="Static loss scale, positive power of 2 values can improve fp16 convergence."
    ) = 1,
    prof: Integer(description="Run only N iterations") = -1,
    seed: Integer(description="random seed used for numpy and pytorch") = None,
    raport_file: String(
        description="file in which to store JSON experiment raport"
    ) = "experiment_raport.json",
    workspace: String(
        description="path to directory where checkpoints will be stored"
    ) = "./",
    save_checkpoint_epochs: Integer(
        description="how many epochs run between saving checkpoints"
    ) = 10,
):
    new_data_path = Path(train_data).parent / "new_dataset"
    convert_image_directory_to_specific_format(
        image_dir_path=train_data, output_root=new_data_path, is_train=True
    )
    convert_image_directory_to_specific_format(
        image_dir_path=val_data, output_root=new_data_path
    )
    print(f"new data path {new_data_path}")
    sys.argv = [
        "main",
        "--data",
        str(new_data_path),
        "--data-backend",
        data_backend.value,
        "--arch",
        arch.value,
        "--model-config",
        model_config.value,
        "-j",
        str(workers),
        "--epochs",
        str(epochs),
        "-b",
        str(batch_size),
        "--optimizer-batch-size",
        str(optimizer_batch_size),
        "--lr",
        str(lr),
        "--lr-schedule",
        lr_schedule.value,
        "--warmup",
        str(warmup),
        "--label-smoothing",
        str(label_smoothing),
        "--mixup",
        str(mixup),
        "--momentum",
        str(momentum),
        "--weight-decay",
        str(weight_decay),
        "--print-freq",
        str(print_freq),
        "--resume",
        str(resume),
        "--pretrained-weights",
        str(pretrained_weights),
        "--static-loss-scale",
        str(static_loss_scale),
        "--prof",
        str(prof),
        "--seed",
        str(seed),
        "--raport-file",
        str(raport_file),
        "--workspace",
        str(workspace),
        "--save-checkpoint-epochs",
        str(save_checkpoint_epochs),
    ]
    print(" ".join(sys.argv))
    runpy.run_path("main.py", run_name="__main__")
