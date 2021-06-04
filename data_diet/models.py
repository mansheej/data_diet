from flax import linen as nn
from functools import partial
from jax import numpy as jnp
from jax.tree_util import tree_flatten
from typing import Any, Callable, Sequence, Tuple
import numpy as np


########################################################################################################################
#  SimpleCNN
########################################################################################################################

class SimpleCNN(nn.Module):
  num_channels: Sequence[int]
  num_classes: int
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x, train=False):  # train is a dummy argument, model does not have different train and eval modes
    for nc in self.num_channels:
      x = nn.Conv(nc, (3, 3), padding='SAME', dtype=self.dtype)(x)
      x = nn.relu(x)
      x = nn.Conv(nc, (3, 3), (2, 2), 'SAME', dtype=self.dtype)(x)
      x = nn.relu(x)
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
    return x


########################################################################################################################
#  ResNet18: based on flax and elegy implementations of ResNet V1
########################################################################################################################

ModuleDef = Any


class ResNetBlock(nn.Module):
  """ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x,):
    residual = x
    y = self.conv(self.filters, (3, 3), self.strides)(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3))(y)
    y = self.norm()(y)

    if residual.shape != y.shape:
      residual = self.conv(self.filters, (1, 1), self.strides, name='conv_proj')(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
  """Bottleneck ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    residual = x
    y = self.conv(self.filters, (1, 1))(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3), self.strides)(y)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters * 4, (1, 1))(y)
    y = self.norm(scale_init=nn.initializers.zeros)(y)

    if residual.shape != y.shape:
      residual = self.conv(self.filters * 4, (1, 1),
                           self.strides, name='conv_proj')(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class ResNet(nn.Module):
  """ResNetV1."""
  stage_sizes: Sequence[int]
  block_cls: ModuleDef
  num_classes: int
  num_filters: int = 64
  lowres: bool = True
  dtype: Any = jnp.float32
  act: Callable = nn.relu

  @nn.compact
  def __call__(self, x, train: bool = True):
    conv = partial(nn.Conv, use_bias=False, dtype=self.dtype)
    norm = partial(nn.BatchNorm, use_running_average=not train, momentum=0.9, epsilon=1e-5, dtype=self.dtype)

    x = conv(self.num_filters,
             (3, 3) if self.lowres else (7, 7),
             (1, 1) if self.lowres else (2, 2),
             padding='SAME',
             name='conv_init')(x)
    x = norm(name='bn_init')(x)
    x = nn.relu(x)
    if not self.lowres:
      x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_cls(self.num_filters * 2 ** i, strides=strides, conv=conv, norm=norm, act=self.act)(x)
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
    x = jnp.asarray(x, self.dtype)
    return x


ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock)
ResNet50 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock)


########################################################################################################################
#  Utils
########################################################################################################################

def get_model(args):
  if args.model == 'resnet18_lowres':
    model = ResNet18(num_classes=args.num_classes, lowres=True)
  elif args.model == 'resnet50_lowres':
    model = ResNet50(num_classes=args.num_classes, lowres=True)
  elif args.model == 'simple_cnn_0':
    model = SimpleCNN(num_channels=[32, 64, 128], num_classes=args.num_classes)
  else:
    raise NotImplementedError
  return model


def get_num_params(params):
  return int(sum([np.prod(w.shape) for w in tree_flatten(params)[0]]))


def get_apply_fn_test(model):
  def apply_fn_test(params, model_state, x):
    vs = {'params': params, **model_state}
    logits = model.apply(vs, x, train=False, mutable=False)
    return logits
  return apply_fn_test


def get_apply_fn_train(model):
  def apply_fn_train(params, model_state, x):
    vs = {'params': params, **model_state}
    logits, model_state = model.apply(vs, x, mutable=list(model_state.keys()))
    return logits, model_state
  return apply_fn_train
