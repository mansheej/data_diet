from .data import test_batches
from .metrics import cross_entropy_loss, accuracy


def get_test_step(f_test):
  def test_step(state, x, y):
    logits = f_test(state.optim.target, state.model, x)
    loss = cross_entropy_loss(logits, y)
    acc = accuracy(logits, y)
    return loss, acc, logits
  return test_step


def test(test_step, state, X, Y, batch_size):
  loss, acc, N = 0, 0, X.shape[0]
  for n, x, y in test_batches(X, Y, batch_size):
    step_loss, step_acc, _ = test_step(state, x, y)
    loss += step_loss * n
    acc += step_acc * n
  loss, acc = loss.item()/N, acc.item()/N
  return loss, acc
