import json
import numpy as np
import os
import shutil
import tensorflow as tf
from types import SimpleNamespace
from .models import get_apply_fn_test, get_model
from .train_state import get_train_state


########################################################################################################################
#  Args
########################################################################################################################

def save_args(args, save_dir, verbose=True):
  save_path = save_dir + '/args.json'
  with open(save_path, 'w') as f: json.dump(vars(args), f, indent=4)
  if verbose: print(f'Save args to {save_path}')
  return save_dir + '/args.json'


def load_args(load_dir, verbose=True):
  load_path = load_dir + '/args.json'
  with open(load_path, 'r') as f: args = SimpleNamespace(**json.load(f))
  if verbose: print(f'Load args from {load_path}')
  return args


def print_args(args):
  print(json.dumps(vars(args), indent=4))


########################################################################################################################
#  File Management
########################################################################################################################

def make_dir(path):
  if os.path.exists(path):
    shutil.rmtree(path)
  os.makedirs(path)


########################################################################################################################
#  Models
########################################################################################################################

def get_fn_params_state(args):
  model = get_model(args)
  fn = get_apply_fn_test(model)
  state, args = get_train_state(args, model)
  params, state = state.optim.target, state.model
  return fn, params, state


########################################################################################################################
#  Seed
########################################################################################################################

def set_global_seed(seed=0):
  np.random.seed(seed)
  tf.random.set_seed(seed)
