import numpy as np
from types import SimpleNamespace


def init_forget_stats(args):
  forget_stats = SimpleNamespace()
  forget_stats.prev_accs = np.zeros(args.num_train_examples, dtype=np.int32)
  forget_stats.num_forgets = np.zeros(args.num_train_examples, dtype=float)
  forget_stats.never_correct = np.arange(args.num_train_examples, dtype=np.int32)
  return forget_stats


def update_forget_stats(forget_stats, idxs, accs):
  forget_stats.num_forgets[idxs[forget_stats.prev_accs[idxs] > accs]] += 1
  forget_stats.prev_accs[idxs] = accs
  forget_stats.never_correct = np.setdiff1d(forget_stats.never_correct, idxs[accs.astype(bool)], True)
  return forget_stats


def save_forget_scores(save_dir, ckpt, forget_stats):
  forget_scores = forget_stats.num_forgets.copy()
  forget_scores[forget_stats.never_correct] = np.inf
  np.save(save_dir + f'/forget_scores/ckpt_{ckpt}.npy', forget_scores)


def load_forget_scores(load_dir, ckpt):
  return np.load(load_dir + f'/forget_scores/ckpt_{ckpt}.npy')
