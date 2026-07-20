import sys, os, time, itertools, traceback
import numpy as np
sys.path.insert(0, os.path.expanduser('~/discourse/foundation'))
sys.path.insert(0, os.path.expanduser('~/discourse/runs/stack'))
import ev
from common import TYPES
from textfeat import TITLE_TR,FORUM_TR,TITLE_TE,FORUM_TE
from fast_score import FastScorer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
BASE=os.path.expanduser('~/discourse')
D=np.load(BASE+'/runs/stack/feats.npz')
Xs_tr,Xs_te=D['Xtr'],D['Xte']; ytr=D['ytr']; folds=D['folds_tr']
prev_tr,next_tr=D['prev_tr'],D['next_tr']; prev_te,next_te=D['prev_te'],D['next_te']
NT=5; W=np.array([0.25,0.375,0.375])
def load3(s):
    return [np.clip(np.load(f'{BASE}/runs/{r}/{s}.npy'),1e-9,1).astype(np.float64) for r in ('gbm','textlin','nnseq')]
g_o,t_o,n_o=load3('oof_probs'); g_t,t_t,n_t=load3('test_probs')
def blend(g,t,n):
    P=np.exp(W[0]*np.log(g)+W[1]*np.log(t)+W[2]*np.log(n));return P/P.sum(1,keepdims=True)
Bo=blend(g_o,t_o,n_o);Bt=blend(g_t,t_t,n_t)
def neigh(B,pv,nx):
    n=B.shape[0];p=np.zeros((n,NT));q=np.zeros((n,NT));m=pv>=0;p[m]=B[pv[m]];m2=nx>=0;q[m2]=B[nx[m2]];return p,q
pvo,nxo=neigh(Bo,prev_tr,next_tr);pvt,nxt_=neigh(Bt,prev_te,next_te)
def geo(*mw):
    L=sum(w*np.log(np.clip(M,1e-9,1)) for M,w in mw);P=np.exp(L);return P/P.sum(1,keepdims=True)
def dense(g,t,n,B,pv,nx,Xs):
    return np.concatenate([np.log(np.concatenate([g,t,n],1)),np.log(B),np.log(np.clip(pv,1e-9,1)),np.log(np.clip(nx,1e-9,1)),Xs],1)
Dtr=dense(g_o,t_o,n_o,Bo,pvo,nxo,Xs_tr); Dte=dense(g_t,t_t,n_t,Bt,pvt,nxt_,Xs_te)
def vecs():
    return [TfidfVectorizer(analyzer='word',ngram_range=(1,2),min_df=2,sublinear_tf=True),
            TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=3,sublinear_tf=True),
            TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),min_df=2,sublinear_tf=True)]
def fit_text(vs,idx): vs[0].fit(TITLE_TR[idx]);vs[1].fit(TITLE_TR[idx]);vs[2].fit(FORUM_TR[idx]);return vs
def tf(vs,ti,fo): return sparse.hstack([vs[0].transform(ti),vs[1].transform(ti),vs[2].transform(fo)]).tocsr()
def oof_lr(C,cw):
    oof=np.zeros((len(Dtr),NT)); test=np.zeros((len(Dte),NT))
    for f in range(5):
        trm=np.where(folds!=f)[0]; prm=np.where(folds==f)[0]
        sc=StandardScaler().fit(Dtr[trm]); vs=fit_text(vecs(),trm)
        Xtr=sparse.hstack([sparse.csr_matrix(sc.transform(Dtr[trm])),tf(vs,TITLE_TR[trm],FORUM_TR[trm])]).tocsr()
        Xpr=sparse.hstack([sparse.csr_matrix(sc.transform(Dtr[prm])),tf(vs,TITLE_TR[prm],FORUM_TR[prm])]).tocsr()
        m=LogisticRegression(C=C,class_weight=cw,max_iter=4000,tol=1e-3); m.fit(Xtr,ytr[trm]); oof[prm]=m.predict_proba(Xpr)
    sc=StandardScaler().fit(Dtr); vs=fit_text(vecs(),np.arange(len(Dtr)))
    Xtr=sparse.hstack([sparse.csr_matrix(sc.transform(Dtr)),tf(vs,TITLE_TR,FORUM_TR)]).tocsr()
    Xte=sparse.hstack([sparse.csr_matrix(sc.transform(Dte)),tf(vs,TITLE_TE,FORUM_TE)]).tocsr()
    m=LogisticRegression(C=C,class_weight=cw,max_iter=4000,tol=1e-3); m.fit(Xtr,ytr); test=m.predict_proba(Xte)
    return oof,test
rng=np.random.RandomState(0); half=rng.rand(len(ytr))<0.5
def sc_mask(P,mask,mode='viterbi'):
    seqs=ev._DEC.decode(np.clip(P,1e-9,1),mode=mode); idx=np.where(mask)[0]
    fs=FastScorer([ev._true_types[i] for i in idx],[ev._true_pars[i] for i in idx],[ev._pred_pars[i] for i in idx])
    return fs.score([np.array([TYPES.index(t) for t in seqs[i]]) for i in idx])[0]
store={'Bo':Bo,'Bt':Bt}
res={}
for C in [0.5,0.7,1.0,1.5]:
    t0=time.time(); o,te=oof_lr(C,'balanced')
    store[f'oof_C{C}']=o; store[f'test_C{C}']=te
    np.savez(BASE+'/runs/stack/meta4_out.npz',**store)  # checkpoint each
    sv,_=ev.score_matrix(o,'viterbi'); sp,_=ev.score_matrix(o,'posterior')
    a=sc_mask(o,half); b=sc_mask(o,~half)
    res[C]=(o,te,sv,sp)
    print(f'C={C}: vit {sv:.4f} post {sp:.4f} | halfA {a:.4f} halfB {b:.4f} ({time.time()-t0:.0f}s)',flush=True)
try:
    bestC=max(res,key=lambda k:max(res[k][2],res[k][3]))
    lo,lt,sv,sp=res[bestC]; mode='posterior' if sp>sv else 'viterbi'
    print('bestC',bestC,'mode',mode,flush=True)
    bb=(-1,None)
    for w in [i/16 for i in range(17)]:
        M=geo((lo,w),(Bo,1-w))
        for md in ('viterbi','posterior'):
            s,_=ev.score_matrix(M,md)
            if s>bb[0]: bb=(s,(w,md))
    w,md=bb[1]; M=geo((lo,w),(Bo,1-w))
    print('blend-back best w=%.3f md=%s full=%.4f A=%.4f B=%.4f'%(w,md,bb[0],sc_mask(M,half,md),sc_mask(M,~half,md)),flush=True)
    Mo=M; Mt=geo((lt,w),(Bt,1-w))
    s,c=ev.score_matrix(Mo,md)
    print('type',{k:round(v,3) for k,v in c['type_percls'].items()})
    print('anchor',{k:round(v,3) for k,v in c['anchor_percls'].items()})
    store.update(dict(lr_oof=lo,lr_test=lt,Mo=Mo,Mt=Mt,bb_w=w,bb_mode=md,bestC=bestC,meta_mode=mode))
    np.savez(BASE+'/runs/stack/meta4_out.npz',**store)
    print('saved meta4_out.npz')
except Exception:
    traceback.print_exc()
