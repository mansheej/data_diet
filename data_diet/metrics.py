from flax import linen as nn
from jax import numpy as jnp


def cross_entropy_loss(logits, labels):
  return jnp.mean(-jnp.sum(nn.log_softmax(logits) * labels, axis=-1))


def correct(logits, labels):
  return jnp.argmax(logits, axis=-1) == jnp.argmax(labels, axis=-1)


def accuracy(logits, labels):
  return jnp.mean(correct(logits, labels))
