import numpy as np, datetime, csv
from paths import log_files, screenshots, require_data
from tonal3 import series, endtime

# ---------- CSV-based detector ----------
def load_series():
    rows=[]
    for f in log_files():
        for r in csv.DictReader(open(f)):
            try:
                rows.append((datetime.datetime.fromisoformat(r['time']),
                             float(r['LAeq']),float(r['LA90']),float(r['LA10'])))
            except Exception: pass
    rows.sort()
    t0=rows[0][0].replace(second=0,microsecond=0)
    n=int((rows[-1][0]-t0).total_seconds()//60)+1
    laeq=np.full(n,np.nan); l90=np.full(n,np.nan); l10=np.full(n,np.nan)
    for t,a,b,c in rows:
        i=int((t-t0).total_seconds()//60)
        if 0<=i<n: laeq[i],l90[i],l10[i]=a,b,c
    return t0,laeq,l90,l10

def rolling_pct(x,win,q):
    n=len(x); out=np.full(n,np.nan); h=win//2
    for i in range(n):
        s=max(0,i-h); e=min(n,i+h+1)
        w=x[s:e]; w=w[~np.isnan(w)]
        if len(w)>=20: out[i]=np.percentile(w,q)
    return out

def detect(t0,l90,thr=3.5,minlen=4,maxgap=3,maxlen=75):
    base=rolling_pct(l90,91,20)
    exc=l90-base
    on=(exc>=thr)&~np.isnan(exc)
    ev=[]; i=0; n=len(on)
    while i<n:
        if on[i]:
            j=i
            gap=0; last=i
            while j<n:
                if on[j]: last=j; gap=0
                else:
                    gap+=1
                    if gap>maxgap: break
                j+=1
            if last-i+1>=minlen and last-i+1<=maxlen:
                ev.append((i,last))
            i=last+1
        else: i+=1
    return ev, exc, base

# ---------- spectral labels from screenshots ----------
def spectral_labels():
    lab={}
    for p in screenshots():
        try:
            H,LO=series(p); end=endtime(p)
        except Exception as e:
            print('skip',p,e); continue
        m=H-LO
        v=m[~np.isnan(m)]
        if len(v)<100: continue
        thr=np.percentile(v,25)+8.0     # per-image adaptive
        for i in range(480):
            if np.isnan(m[i]): continue
            t=end-datetime.timedelta(seconds=180*(479-i))
            lab[t.replace(second=0,microsecond=0)]=(m[i],m[i]>thr)
    return lab

if __name__=='__main__':
    require_data()
    t0,laeq,l90,l10=load_series()
    ev,exc,base=detect(t0,l90)
    print('candidate events:',len(ev))
    lab=spectral_labels()
    print('spectral-labelled 3-min slots:',len(lab))
    # evaluate: for each labelled slot, is it inside a detected event?
    det=np.zeros(len(l90),bool)
    for a,b in ev: det[a:b+1]=True
    tp=fp=fn=tn=0
    for t,(val,isfan) in lab.items():
        i=int((t-t0).total_seconds()//60)
        if not (0<=i<len(det)) or np.isnan(l90[i]): continue
        d=bool(det[max(0,i-1):i+2].any())
        if isfan and d: tp+=1
        elif isfan and not d: fn+=1
        elif (not isfan) and d: fp+=1
        else: tn+=1
    print(f'TP {tp}  FP {fp}  FN {fn}  TN {tn}')
    if tp+fp: print('precision %.2f'%(tp/(tp+fp)))
    if tp+fn: print('recall    %.2f'%(tp/(tp+fn)))
    np.save('exc.npy',exc); np.save('l90.npy',l90); np.save('laeq.npy',laeq)
    import pickle; pickle.dump((t0,ev,lab),open('state.pkl','wb'))
