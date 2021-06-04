from jax import jacrev, jit, vmap
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_map
import numpy as np
from tqdm import tqdm


def flatten_jacobian(J):
  """Jacobian pytree -> Jacobian matrix"""
  return jnp.concatenate(tree_flatten(tree_map(vmap(jnp.ravel), J))[0], axis=1)


def get_mean_logit_gradients_fn(fn, params, state):
  """fn, params, state -> (X -> mean logit gradients of fn(X; params, state))"""
  return lambda X: flatten_jacobian(jacrev(lambda p, x: fn(p, state, x).mean(0))(params, X))


def compute_mean_logit_gradients(fn, params, state, X, batch_sz):
  """compute_mean_logit_gradients: fn, params, state, X, batch_sz -> mlg
  In:
    fn      : func           : params, state, X -> logits of X at (params, state)
    params  : pytree         : parameters
    state   : pytree         : model state
    X       : nparr(n, image): images
    batch_sz: int            : image batch size for computation
  Out:
    mlgs: nparr(n_cls, n_params): mean logit gradients
  """
  # batch data for computation
  n_batches = X.shape[0] // batch_sz
  Xs = np.split(X, n_batches)
  # compute mean logit gradients
  mean_logit_gradients = jit(get_mean_logit_gradients_fn(fn, params, state))
  mlgs = 0
  for X in tqdm(Xs):
    mlgs += np.array(mean_logit_gradients(X)) / n_batches
  return mlgs
