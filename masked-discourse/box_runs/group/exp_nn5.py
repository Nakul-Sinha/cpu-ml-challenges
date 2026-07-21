import sys, os, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser('~/solcheck'))
sys.path.insert(0, os.path.expanduser('~/discourse/foundation'))
import solution as S

t0 = time.time()
ROOT = os.path.expanduser('~/solcheck/dataset/public')
train = pd.read_csv(ROOT + '/train.csv')
test = pd.read_csv(ROOT + '/test.csv')
fold_rows = np.load(os.path.expanduser('~/discourse/runs/group_folds.npy'))
n_oof, n_test, nm = S.run_nnseq(train, test, fold_rows, 5)
np.save(os.path.expanduser('~/discourse/runs/group_n5_oof.npy'), n_oof)
np.save(os.path.expanduser('~/discourse/runs/group_n5_test.npy'), n_test)
print(f'nn5 done {nm} models {time.time()-t0:.0f}s', flush=True)
