import numpy as np, datetime, csv, collections
from paths import EVENTS_CSV, screenshots, require_data
from detect import load_series, detect
from refine import shape_ok
from tonal3 import series, endtime

require_data()
t0,laeq,l90,l10=load_series()
ev,exc,base=detect(t0,l90,thr=4.0,minlen=4,maxlen=60)
events=[]
for a,b in ev:
    s=shape_ok(l90,a,b)
    if s: events.append((a,b,s))

# spectral metric per 3-min slot
spec={}
for p in screenshots():
    try: H,LO=series(p); end=endtime(p)
    except Exception as e: print('skip',p.name,e); continue
    m=H-LO; v=m[~np.isnan(m)]
    if len(v)<100: continue
    thr=np.percentile(v,25)+8.0
    for i in range(480):
        if np.isnan(m[i]): continue
        t=(end-datetime.timedelta(seconds=180*(479-i))).replace(second=0,microsecond=0)
        prev=spec.get(t)
        val=bool(m[i]>thr)
        spec[t]=val if prev is None else (prev or val)

def confirm(a,b):
    hits=tot=0
    for k in range(a,b+1):
        t=(t0+datetime.timedelta(minutes=k)).replace(second=0,microsecond=0)
        for d in (-1,0,1):
            tt=t+datetime.timedelta(minutes=d)
            if tt in spec:
                tot+=1; hits+= spec[tt]; break
    if tot==0: return 'no_screenshot'
    return 'confirmed' if hits/tot>=0.4 else 'not_confirmed'

EQGAP_MAX=2.0; SPREAD_MAX=3.5
rows=[]; dropped=0
for a,b,s in events:
    ta=t0+datetime.timedelta(minutes=a); tb=t0+datetime.timedelta(minutes=b)
    seg_eq=laeq[a:b+1]; seg_9=l90[a:b+1]; seg_1=l10[a:b+1]
    m=~np.isnan(seg_eq)&~np.isnan(seg_9)&~np.isnan(seg_1)
    if m.sum()<4:
        dropped+=1; continue
    eqgap=float(np.median(seg_eq[m]-seg_9[m])); spread=float(np.median(seg_1[m]-seg_9[m]))
    if eqgap>EQGAP_MAX or spread>SPREAD_MAX:      # fluctuating source (speech/radio/TV), not the fan
        dropped+=1; continue
    seg_eq=seg_eq[~np.isnan(seg_eq)]
    rows.append(dict(
        date=ta.strftime('%Y-%m-%d'), start=ta.strftime('%H:%M'), end=tb.strftime('%H:%M'),
        duration_min=s['dur'],
        L90_during=round(s['level'],1),
        L90_before=round(s['level']-s['rise'],1),
        rise_dB=round(s['rise'],1),
        LAeq_mean=round(float(np.mean(seg_eq)),1) if len(seg_eq) else '',
        LAeq_max=round(float(np.max(seg_eq)),1) if len(seg_eq) else '',
        LAeq_minus_L90=round(eqgap,1),
        LA10_minus_L90=round(spread,1),
        spectral=confirm(a,b)))
rows.sort(key=lambda r:(r['date'],r['start']))
out=EVENTS_CSV
with open(out,'w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('wrote',out,len(rows),'events; dropped',dropped,'as fluctuating/short')
c=collections.Counter(r['spectral'] for r in rows); print(c)
# summary
pre=[r for r in rows if r['date']<='2026-09-17']; post=[r for r in rows if r['date']>'2026-09-17']
print('events to Sep 17:',len(pre),' after:',len(post))
print('median duration %.0f min'%np.median([r['duration_min'] for r in pre]))
print('median L90 during %.1f, before %.1f, rise %.1f'%(np.median([r['L90_during'] for r in pre]),np.median([r['L90_before'] for r in pre]),np.median([r['rise_dB'] for r in pre])))
print('max L90 during %.1f'%max(r['L90_during'] for r in pre))
print('total hours (to Sep 17): %.1f'%(sum(r['duration_min'] for r in pre)/60))
hours=collections.Counter(int(r['start'][:2]) for r in pre)
print('by hour:', ' '.join(f'{h:02d}:{hours.get(h,0)}' for h in range(24)))
conf=[r for r in rows if r['spectral']=='confirmed']
print('spectrally confirmed:',len(conf))
