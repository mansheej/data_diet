# python get_run_score.py <ROOT:str> <EXP:str> <RUN:int> <STEP:int> <BATCH_SZ:int> <TYPE:str>

from data_diet.data import load_data
from data_diet.scores import compute_scores
from data_diet.utils import get_fn_params_state, load_args
import sys
import numpy as np
import os

ROOT = sys.argv[1]
EXP = sys.argv[2]
RUN = int(sys.argv[3])
STEP = int(sys.argv[4])
BATCH_SZ = int(sys.argv[5])
TYPE = sys.argv[6]

run_dir = ROOT + f'/exps/{EXP}/run_{RUN}'
args = load_args(run_dir)
args.load_dir = run_dir
args.ckpt = STEP

_, X, Y, _, _, args = load_data(args)
fn, params, state = get_fn_params_state(args)
scores = compute_scores(fn, params, state, X, Y, BATCH_SZ, TYPE)

path_name = 'error_l2_norm_scores' if TYPE == 'l2_error' else 'grad_norm_scores'

save_dir = run_dir + f'/{path_name}'
save_path = run_dir + f'/{path_name}/ckpt_{STEP}.npy'
if not os.path.exists(save_dir): os.makedirs(save_dir)
np.save(save_path, scores)
