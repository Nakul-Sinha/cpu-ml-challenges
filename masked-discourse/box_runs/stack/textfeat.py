"""Per-node aligned raw text (title, forum) in canonical order, for meta/anchor text models."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser('~/discourse/foundation'))
ROOT=os.path.expanduser('~/discourse/dataset/public')
def per_node_text(df):
    titles=[]; forums=[]
    for _,r in df.iterrows():
        L=len(r['masked_nodes'].split())
        for _ in range(L):
            titles.append(str(r['thread_title'])); forums.append(str(r['forum']))
    return np.array(titles,dtype=object), np.array(forums,dtype=object)
train=pd.read_csv(os.path.join(ROOT,'train.csv')); test=pd.read_csv(os.path.join(ROOT,'test.csv'))
TITLE_TR,FORUM_TR=per_node_text(train); TITLE_TE,FORUM_TE=per_node_text(test)
