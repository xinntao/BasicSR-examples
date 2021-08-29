# :rocket: BasicSR Examples

[![download](https://img.shields.io/github/downloads/xinntao/BasicSR-examples/total.svg)](https://github.com/xinntao/BasicSR-examples/releases)
[![Open issue](https://img.shields.io/github/issues/xinntao/BasicSR-examples)](https://github.com/xinntao/BasicSR-examples/issues)
[![Closed issue](https://img.shields.io/github/issues-closed/xinntao/BasicSR-examples)](https://github.com/xinntao/BasicSR-examples/issues)
[![LICENSE](https://img.shields.io/github/license/xinntao/basicsr-examples.svg)](https://github.com/xinntao/BasicSR-examples/blob/master/LICENSE)
[![python lint](https://github.com/xinntao/BasicSR/actions/workflows/pylint.yml/badge.svg)](https://github.com/xinntao/BasicSR/blob/master/.github/workflows/pylint.yml)

[English](README.md) **|** [简体中文](README_CN.md) <br>
[`BasicSR repo`](https://github.com/xinntao/BasicSR) **|** [`simple mode example`](https://github.com/xinntao/BasicSR-examples/tree/master) **|** [`installation mode example`](https://github.com/xinntao/BasicSR-examples/tree/installation)

In this repository, we give examples to illustrate **how to easily use** [`BasicSR`](https://github.com/xinntao/BasicSR) in **your own project**.

:triangular_flag_on_post: **Projects that use BasicSR**
- :white_check_mark: [**GFPGAN**](https://github.com/TencentARC/GFPGAN): A practical algorithm for real-world face restoration
- :white_check_mark: [**Real-ESRGAN**](https://github.com/xinntao/Real-ESRGAN): A practical algorithm for general image restoration

If you use `BasicSR` in your open-source projects, welcome to contact me (by [email](#e-mail-contact) or opening an issue/pull request). I will add your projects to the above list :blush:

---

If this repo is helpful, please help to :star: this repo or recommend it to your friends. Thanks:blush: <br>
Other recommended projects:<br>
:arrow_forward: [facexlib](https://github.com/xinntao/facexlib): A collection that provides useful face-relation functions.<br>
:arrow_forward: [HandyView](https://github.com/xinntao/HandyView): A PyQt5-based image viewer that is handy for view and comparison.

---

## Contents

- [HowTO use BasicSR](#HowTO-use-BasicSR)
- [As s Template](#As-a-Template)

## HowTO use BasicSR

`BasicSR` can be used in two ways:
- :arrow_right: Git clone the entire BasicSR. In this way, you can see the complete codes of BasicSR, and then modify them according to your own needs.
- :arrow_right: Use basicsr as a [python package](https://pypi.org/project/basicsr/#history) (that is, install with pip). It provides the training framework, procedures, and some basic functions. You can easily build your own projects based on basicsr.
    ```bash
    pip install basicsr
    ```

Our example mainly focuses on the second one, that is, how to easily and concisely build your own project based on the basicsr package.

There are two ways to use the python package of basicsr, which are provided in two branches:

- :arrow_right: [simple mode](https://github.com/xinntao/BasicSR-examples/tree/master): the project can be run **without installation**. But it has limitations: it is inconvenient to import complex hierarchical relationships; It is not easy to access the functions in this project from other locations

- :arrow_right: [installation mode](https://github.com/xinntao/BasicSR-examples/tree/installation): you need to install the project by running `python setup.py develop`. After installation, it is more convenient to import and use.

As a simple introduction and explanation, we use the example of *simple mode*, but we recommend the *installation mode* in practical use.

```bash
git clone https://github.com/xinntao/BasicSR-examples.git
cd BasicSR-examples
```

### Preliminary

Most deep-learning projects can be divided into the following parts:

1. **data**: defines the training/validation data that is fed into the model training
2. **arch** (architecture): defines the network structure and the forward steps
3. **model**: defines the necessary components in training (such as loss) and a complete training process (including forward propagation, back-propagation, gradient optimization, *etc*.), as well as other functions, such as validation, *etc*
4. Training pipeline: defines the training process, that is, connect the data-loader, model, validation, saving checkpoints, *etc*

When we are developing a new method, we often improve the **data**, **arch**, and **model**. Most training processes and basic functions are actually shared. Then, we hope to focus on the development of main functions instead of building wheels repeatedly.

Therefore, we have BasicSR, which separates many shared functions. With BasicSR, we just need to care about the development of **data**, **arch**, and **model**.

In order to further facilitate the use of BasicSR, we provide the basicsr package. You can easily install it through `pip install basicsr`. After that, you can use the training process of BasicSR and the functions already developed in BasicSR~

### A Simple Example

Let's use a simple example to illustrate how to use BasicSR to build your own project.

We provide two sample data for demonstration:
1. [BSDS100](https://github.com/xinntao/BasicSR-examples/releases/download/0.0.0/BSDS100.zip) for training
1. [Set5](https://github.com/xinntao/BasicSR-examples/releases/download/0.0.0/Set5.zip) for validation

You can easily download them by running the following command in the BasicSR-examples root path:

```bash
python scripts/prepare_example_data.py
```

The sample data are now in the `datasets/example` folder.

#### :zero: Purpose

Let's use a Super-Resolution task for the demo.
It takes a low-resolution image as the input and outputs a high-resolution image.
The low-resolution images contain: 1) CV2 bicubic X4 downsampling, and 2) JPEG compression (quality = 70).

In order to better explain how to use the arch and model, we use 1) a network structure similar to SRCNN; 2) use L1 and L2 (MSE) loss simultaneously in training.

So, in this task, what we should do are:

1. Build our own data loader
1. Determine the architecture
1. Build our own model

Let's explain it separately in the following parts.

#### :one: data

We need to implement a new dataset to fulfill our purpose. The dataset is used to feed the data into the model.

An example of this dataset is in [data/example_dataset.py](data/example_dataset.py). It has the following steps.

1. Read Ground-Truth (GT) images. BasicSR provides [FileClient](https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/file_client.py) for easily reading files in a folder, LMDB file and meta_info txt. In this example, we use the folder mode. For more reading modes, please refer to [basicsr/data](https://github.com/xinntao/BasicSR/tree/master/basicsr/data)
1. Synthesize low resolution images. We can directly implement the data procedures in the `__getitem__(self, index)` function, such as downsampling and adding JPEG compression. Many basic operations can be found in [[basicsr/data/degradations]](https://github.com/xinntao/BasicSR/blob/master/basicsr/data/degradations.py), [[basicsr/data/tranforms]](https://github.com/xinntao/BasicSR/blob/master/basicsr/data/transforms.py) ,and [[basicsr/data/data_util]](https://github.com/xinntao/BasicSR/blob/master/basicsr/data/data_util.py)
1. Convert to torch tensor and return appropriate information

**Note**:

1. Please add `@DATASET_REGISTRY.register()` before `ExampleDataset`. This operation is mainly used to prevent the occurrence of a dataset with the same name, which will result in potential bugs
1. The new dataset file should end with `_dataset.py`, such as `example_dataset.py`. In this way, the program can **automatically** import classes without manual import

In the [option configuration file](options/example_option.yml), you can use the new dataset:

```yaml
datasets:
  train:  # training dataset
    name: ExampleBSDS100
    type: ExampleDataset  # the class name

    # ----- the followings are the arguments of ExampleDataset ----- #
    dataroot_gt: datasets/example/BSDS100
    io_backend:
      type: disk

    gt_size: 128
    use_flip: true
    use_rot: true

    # ----- arguments of data loader ----- #
    use_shuffle: true
    num_worker_per_gpu: 3
    batch_size_per_gpu: 16
    dataset_enlarge_ratio: 10
    prefetch_mode: ~

  val:  # validation dataset
    name: ExampleSet5
    type: ExampleDataset
    dataroot_gt: datasets/example/Set5
    io_backend:
      type: disk
```

#### :two: arch

An example of architecture is in [archs/example_arch.py](archs/example_arch.py). It mainly builds the network structure.

**Note**:

1. Add `@ARCH_REGISTRY.register()` before `ExampleArch`, so as to register the newly implemented arch. This operation is mainly used to prevent the occurrence of arch with the same name, resulting in potential bugs
1. The new arch file should end with `_arch.py`, such as `example_arch.py`. In this way, the program can **automatically** import classes without manual import

In the [option configuration file](options/example_option.yml), you can use the new arch:

```yaml
# network structures
network_g:
  type: ExampleArch  # the class name

  # ----- the followings are the arguments of ExampleArch ----- #
  num_in_ch: 3
  num_out_ch: 3
  num_feat: 64
  upscale: 4
```

#### :three: model

An example of model is in [models/example_model.py](models/example_model.py). It mainly builds the training process of a model.

In this file:
1. We inherit `SRModel` from basicsr. Many models have similar operations, so you can inherit and modify from [basicsr/models](https://github.com/xinntao/BasicSR/tree/master/basicsr/models). In this way, you can easily implement your ideas, such as GAN model, video model, *etc*.
1. Two losses are used: L1 and L2 (MSE) loss
1. Many other contents, such as `setup_optimizers`, `validation`, `save`, *etc*, are inherited from `SRModel`

**Note**:

1. Add `@MODEL_REGISTRY.register()` before `ExampleModel`, so as to register the newly implemented model. This operation is mainly used to prevent the occurrence of model with the same name, resulting in potential bugs
1. The new model file should end with `_model.py`, such as `example_model.py`. In this way, the program can **automatically** import classes without manual import

In the [option configuration file](options/example_option.yml), you can use the new model:

```yaml
# training settings
train:
  optim_g:
    type: Adam
    lr: !!float 2e-4
    weight_decay: 0
    betas: [0.9, 0.99]

  scheduler:
    type: MultiStepLR
    milestones: [50000]
    gamma: 0.5

  total_iter: 100000
  warmup_iter: -1  # no warm up

  # ----- the followings are the configurations for two losses ----- #
  # losses
  l1_opt:
    type: L1Loss
    loss_weight: 1.0
    reduction: mean

  l2_opt:
    type: MSELoss
    loss_weight: 1.0
    reduction: mean
```

#### :four: training pipeline

The whole training pipeline can reuse the [basicsr/train.py](https://github.com/xinntao/BasicSR/blob/master/basicsr/train.py) in BasicSR.

Based on this, our [train.py](train.py) can be very concise:

```python
import os.path as osp

import archs  # noqa: F401
import data  # noqa: F401
import models  # noqa: F401
from basicsr.train import train_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    train_pipeline(root_path)

```

#### :five: debug mode

So far, we have completed the development of our project. We can quickly check whether there is a bug through the `debug` mode:

```bash
python train.py -opt options/example_option.yml --debug
```

With `--debug`, the program will enter the debug mode. In the debug mode, the program will output at each iteration, and perform validation every 8 iterations, so that you can easily know whether the program has a bug~

#### :six: normal training

After debugging, we can have the normal training.

```bash
python train.py -opt options/example_option.yml
```

If the training process is interrupted unexpectedly and the resume is required. Please use `--auto_resume` in the command:

```bash
python train.py -opt options/example_option.yml --auto_resume
```

So far, you have finished developing your own projects using `BasicSR`. Isn't it very convenient~ :grin:

## As a Template

You can use BasicSR-Examples as a template for your project. Here are some modifications you may need.

1. Set up the *pre-commit* hook
    1. In the root path, run:
    > pre-commit install
1. Modify the `LICENSE`<br>
    This repository uses the *MIT* license, you may change it to other licenses

The simple mode do not require many modifications. Those using the installation mode may need more modifications. See [here](https://github.com/xinntao/BasicSR-examples/blob/installation/README.md#As-a-Template)

## :e-mail: Contact

If you have any questions or want to add your project to the list, please email `xintao.wang@outlook.com` or `xintaowang@tencent.com`.
