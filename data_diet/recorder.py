import pickle
from types import SimpleNamespace


def init_recorder():
  rec = SimpleNamespace()
  rec.train_step = []
  rec.train_loss = []
  rec.train_acc = []
  rec.lr = []
  rec.test_step = []
  rec.test_loss = []
  rec.test_acc = []
  rec.ckpts = []
  return rec


def record_train_stats(rec, step, loss, acc, lr):
  rec.train_step.append(step)
  rec.train_loss.append(loss)
  rec.train_acc.append(acc)
  rec.lr.append(lr)
  return rec


def record_test_stats(rec, step, loss, acc):
  rec.test_step.append(step)
  rec.test_loss.append(loss)
  rec.test_acc.append(acc)
  return rec


def record_ckpt(rec, step):
  rec.ckpts.append(step)
  return rec


def save_recorder(save_dir, rec, verbose=True):
  save_path = save_dir + '/recorder.pkl'
  with open(save_path, 'wb') as f: pickle.dump(vars(rec), f)
  if verbose: print(f'Save record to {save_path}')
  return save_path


def load_recorder(load_dir, verbose=True):
  load_path = load_dir + '/recorder.pkl'
  with open(load_path, 'rb') as f: rec = SimpleNamespace(**pickle.load(f))
  if verbose: print(f'Load record from {load_path}')
  return rec
