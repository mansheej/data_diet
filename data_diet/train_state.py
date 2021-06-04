from flax import optim
from flax.struct import dataclass as flax_dataclass
from flax.training import checkpoints
from jax import jit, random
import jax.numpy as jnp
import time
from typing import Any
from .models import get_num_params


@flax_dataclass
class TrainState:
  optim: optim.Optimizer
  model: Any


def create_train_state(args, model):
  @jit
  def init(*args):
    return model.init(*args)
  key, input = random.PRNGKey(args.model_seed), jnp.ones((1, *args.image_shape), model.dtype)
  model_state, params = init(key, input).pop('params')
  if not hasattr(args, 'nesterov'): args.nesterov = False
  opt = optim.Momentum(args.lr, args.beta, args.weight_decay, args.nesterov).create(params)
  train_state = TrainState(optim=opt, model=model_state)
  return train_state


def get_train_state(args, model):
  time_start = time.time()
  print('get train state... ', end='')
  state = create_train_state(args, model)
  if args.load_dir:
    print(f'load from {args.load_dir}/ckpts/checkpoint_{args.ckpt}... ', end='')
    state = checkpoints.restore_checkpoint(args.load_dir + '/ckpts', state, args.ckpt)
  args.num_params = get_num_params(state.optim.target)
  print(f'{int(time.time() - time_start)}s')
  return state, args
