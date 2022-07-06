import importlib
from os import path as osp

from basicsr.utils import scandir

# automatically scan and import loss modules for registry
# scan all the files that end with '_loss.py' under the loss folder
loss_folder = osp.dirname(osp.abspath(__file__))
loss_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(loss_folder) if v.endswith('_loss.py')]
# import all the loss modules
_model_modules = [importlib.import_module(f'losses.{file_name}') for file_name in loss_filenames]
