import numpy as np, datetime, collections
from detect import load_series, detect

def nm(x):
    x=x[~np.isnan(x)]
    return np.median(x) if len(x) else np.nan

def shape_ok(l90,a,b,minstep=3.5,maxstd=2.5,dmin=5,dmax=35):
    d=b-a+1
    if not (dmin<=d<=dmax): return None
    pre=nm(l90[max(0,a-8):max(1,a-1)]); post=nm(l90[b+2:b+10])
    inside=l90[a:b+1]; inside=inside[~np.isnan(inside)]
    if len(inside)<dmin or np.isnan(pre) or np.isnan(post): return None
    lev=np.median(inside)
    rise=lev-pre; fall=lev-post
    if rise<minstep or fall<minstep: return None
    if np.std(inside)>maxstd: return None
    return dict(level=lev,rise=rise,fall=fall,std=float(np.std(inside)),dur=d)

if __name__=='__main__':
    t0,laeq,l90,l10=load_series()
    ev,exc,base=detect(t0,l90,thr=4.0,minlen=4,maxlen=60)
    keep=[]
    for a,b in ev:
        s=shape_ok(l90,a,b)
        if s: keep.append((a,b,s))
    print('events after shape filter:',len(keep),'of',len(ev))
    byday=collections.Counter((t0+datetime.timedelta(minutes=a)).date() for a,b,s in keep)
    pre=post=0
    for d in sorted(byday):
        n=byday[d]
        tag='' if d<=datetime.date(2026,9,17) else '   <-- after Sep 17'
        print(f'  {d} n={n:3d} '+'*'*n+tag)
        if d<=datetime.date(2026,9,17): pre+=n
        else: post+=n
    print(f'total before/on Sep 17: {pre}   after: {post}')
