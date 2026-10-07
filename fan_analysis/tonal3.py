import numpy as np, datetime, re, csv
from paths import SCREENSHOT_DIR, log_files
from PIL import Image
import matplotlib.cm as cm
VIR=(cm.viridis(np.linspace(0,1,256))[:,:3]*255)
Y1K=296.0; DEC=47.0

def vindex(img):
    px=img.reshape(-1,3)
    d=((px[:,None,:]-VIR[None,:,:])**2).sum(-1)
    return d.argmin(1).reshape(img.shape[:2]).astype(float), np.sqrt(d.min(1)).reshape(img.shape[:2])

def series(path, x0=77, x1=631, ncols=480):
    img=np.asarray(Image.open(path).convert('RGB')).astype(float)
    idx,err=vindex(img)
    idx=np.where(err>40, np.nan, idx)
    yf=lambda f: Y1K - DEC*np.log10(f/1000.0)
    sl=lambda a,b: slice(int(round(yf(b))), int(round(yf(a)))+1)
    with np.errstate(all='ignore'):
        hi=np.nanmean(idx[sl(1300,3500), x0:x1+1],axis=0)
        lo=np.nanmean(idx[sl(250,700),  x0:x1+1],axis=0)
    xs=np.arange(x0,x1+1)
    colidx=np.clip(((xs-x0)/(x1-x0)*(ncols-1)).round().astype(int),0,ncols-1)
    H=np.full(ncols,np.nan); L=np.full(ncols,np.nan)
    for c in range(ncols):
        m=colidx==c
        if m.any():
            H[c]=np.nanmedian(hi[m]); L[c]=np.nanmedian(lo[m])
    return H,L

def endtime(p):
    # str(): callers pass Path objects, and re.search on one raises TypeError
    # -- which the callers' except-and-continue would swallow, silently
    # dropping every screenshot and reporting no spectral confirmation at all.
    m=re.search(r'(\d{8})-(\d{6})',str(p))
    if m is None:
        raise ValueError(f'no YYYYMMDD-HHMMSS timestamp in {p}')
    return datetime.datetime.strptime(m.group(1)+m.group(2),'%Y%m%d%H%M%S')

def csv_data():
    d={}
    for f in log_files():
        for r in csv.DictReader(open(f)):
            try:
                t=datetime.datetime.fromisoformat(r['time']).replace(second=0,microsecond=0)
                d[t]=(float(r['LAeq']),float(r['LA90']))
            except Exception: pass
    return d

if __name__=='__main__':
    L=csv_data()
    p=str(SCREENSHOT_DIR/'noise-monitor-20260910-072644.png')
    H,LO=series(p); end=endtime(p)
    times=[end-datetime.timedelta(seconds=180*(479-i)) for i in range(480)]
    m=H-LO
    pair=[]
    for i,t in enumerate(times):
        tt=t.replace(second=0,microsecond=0)
        w=[L[tt+datetime.timedelta(minutes=k)][1] for k in range(-1,2) if tt+datetime.timedelta(minutes=k) in L]
        if w and not np.isnan(m[i]): pair.append((t,m[i],float(np.mean(w))))
    a=np.array([x[1] for x in pair]); b=np.array([x[2] for x in pair])
    print('n',len(pair),'corr(hi-lo, L90)=',round(float(np.corrcoef(a,b)[0,1]),3))
    print('hi-lo percentiles',np.round(np.percentile(a,[10,25,50,75,90]),1))
    for t,mm,l9 in pair:
        if t.date()==datetime.date(2026,9,9) and 12<=t.hour<19:
            print(t.strftime('%m-%d %H:%M'), f'{mm:6.1f} {l9:5.1f} ' + '#'*max(0,int((mm+30))) )
