# BasicSR Examples

[English](README.md) **|** [简体中文](README_CN.md)

[`BasicSR`](https://github.com/xinntao/BasicSR) **|** [`simple example`](https://github.com/xinntao/BasicSR-examples/tree/master) **|** [installation example`](https://github.com/xinntao/BasicSR-examples/tree/installation)

使用 BasicSR 的项目:
:white_check_mark: [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN): Practical Algorithms for General Image Restoration
:white_check_mark: [GFPGAN](https://github.com/TencentARC/GFPGAN): Practical Algorithms for Real-world Face Restoration

如果你的开源项目中使用 BasicSR, 欢迎联系我，将你的开源添加到上面的列表中

---

在这个仓库中，我们提供简单的例子来说明，如何在你自己的项目中使用 [`BasicSR`](https://github.com/xinntao/BasicSR)。<br>
`BasicSR` 有两种使用方式：
:one: Git clone 整个 BasicSR 的代码。这样可以看到 BasicSR 完整的代码，然后根据你自己的需求进行修改。
:two: BasicSR作为一个 [python package](https://pypi.org/project/basicsr/)(即可以通过pip安装)，提供了训练的框架，流程和一些基本功能。你可以基于 BasicSR 方便地搭建你自己的项目。

我们的样例主要针对第二种使用方式，即如何基于 basicsr 这个package来方便简洁地搭建你自己的项目。<br>

使用 basicsr 的python package可以有两种方式，我们分别提供在两个 branch 中:<br>
:one: [简单模式](https://github.com/xinntao/BasicSR-examples/tree/master): 项目的仓库不需要安装，就可以运行使用。但它有局限：不方便 import 复杂的层级关系；在其他位置也不容易访问本项目中的函数
:two: [安装模式](https://github.com/xinntao/BasicSR-examples/tree/installation): 项目的仓库需要安装`python setup.py develop`，安装之后 import和使用都更加方便

作为简单的入门和讲解， 我们使用简单模式的样例，但在实际使用中我们推荐安装模式。

## 预备

大部分的深度学习项目，都可以分为以下几个部分：

1. **data**: 定义了训练数据，来喂给模型的训练过程。
2. **arch** (architecture): 定义了网络结构 和 forward 的步骤。
3. **model**: 定义了在训练中必要的组件（比如 loss） 和 一次完整的训练过程（包括前向传播，反向传播，梯度优化等），还有其他功能，比如 validation等。
4. training pipeline: 定义了训练的流程，即把数据 dataloader，模型，validation，保存checkpoints 等等串联起来。

当我们开发一个新的方法时，我们往往在改进: **data**, **arch**, **model**；而很多流程、基础的功能其实是共用的。那么，我们希望可以专注于主要功能的开发，而不要重复造轮子。<br>
因此便有了 BasicSR，它把很多相似的功能都独立出来，我们只要关心 **data**, **arch**, **model**的开发即可。<br>
为了进一步方便大家使用，我们提供了 basicsr package，大家可以通过 `pip install basicsr` 方便地安装，然后就可以使用 BasicSR 的训练流程以及已经在BasicSR里面开发好的功能啦。

## 简单的例子

下面我们就通过一个简单的例子，来说明如何使用 BasicSR 来搭建你自己的项目。

### 目的

我们来假设一个超分辨率的任务，输入一个低分辨率的图片，输出一个有锐化效果的高分辨率的图片。<br>
在这个任务中，我们要做的是: 构建自己的data loader, architecture 和 model。下面我们分别来说明一下。

### data

### arch

### model

###

debug 模式





In this repository, we give simple examples to illustrate how to use [`BasicSR`](https://github.com/xinntao/BasicSR) in your own project.

## 文件修改

1. 设置 *pre-commit* hook.
    1. 若需要, 修改 `.pre-commit-config.yaml`
    1. 在文件夹根目录, 运行
    > pre-commit install
1. 修改 `.gitignore` 文件
1. 修改 `LICENSE` 文件
    本仓库使用 *MIT* 许可, 根据需要可以修改成其他许可
1. 修改 *setup* 文件
    1. `setup.cfg`
    1. `setup.py`, 特别是其中包含的关键字 `basicsr`
1. 修改 `requirements.txt` 文件
1. 修改 `VERSION` 文件

## GitHub Workflows

1. [pylint](./github/workflows/pylint.yml)
1. [gitee-repo-mirror](./github/workflow/gitee-repo-mirror.yml) - 支持 Gitee码云
    1. 在 [Gitee](https://gitee.com/) 网站克隆 Github 仓库
    1. 修改 [gitee-repo-mirror](./github/workflow/gitee-repo-mirror.yml) 文件
    1. 在 Github 中的 *Settings* -> *Secrets* 的 `SSH_PRIVATE_KEY`

## 其他流程

1. 主页上的 `description`, `website`, `topics`
1. 支持中文文档, 比如, `README_CN.md`

## Emoji

[Emoji cheat-sheet](https://github.com/ikatyang/emoji-cheat-sheet)

| Emoji | Meaning |
| :---         |     :---:      |
| :rocket:   | Used for [BasicSR](https://github.com/xinntao/BasicSR) Logo |
| :sparkles: | Features |
| :zap: | HOWTOs |
| :wrench: | Installation / Usage |
| :hourglass_flowing_sand: | TODO list |
| :turtle: | Dataset preparation |
| :computer: | Commands |
| :european_castle: | Model zoo |
| :memo: | Designs |
| :scroll: | License and acknowledgement |
| :earth_asia: | Citations |
| :e-mail: | Contact |
| :m: | Models |
| :arrow_double_down: | Download |
| :file_folder: | Datasets |
| :chart_with_upwards_trend: | Curves|
| :eyes: | Screenshot |
| :books: |References |

## 有用的图像链接

<img src="https://colab.research.google.com/assets/colab-badge.svg" height="28" alt="google colab logo">  Google Colab Logo <br>
<img src="https://upload.wikimedia.org/wikipedia/commons/8/8d/Windows_darkblue_2012.svg" height="28" alt="google colab logo">  Windows Logo <br>
<img src="https://upload.wikimedia.org/wikipedia/commons/3/3a/Logo-ubuntu_no%28r%29-black_orange-hex.svg" alt="Ubuntu" height="24">  Ubuntu Logo <br>

## 其他有用的技巧

1. `More` 下拉菜单
    <details>
    <summary>More</summary>
    <ul>
    <li>Nov 19, 2020. Set up ProjectTemplate-Python.</li>
    </ul>
    </details>
