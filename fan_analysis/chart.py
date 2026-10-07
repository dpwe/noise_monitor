import numpy as np, datetime, csv, json, urllib.request, os
from paths import EVENTS_CSV, HERE, log_files, require_data
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

SURF='#fcfcfb'; INK='#0b0b0b'; INK2='#52514e'; MUTED='#8b8a85'
DOC_WIDTH_PX=1300          # fan_doc.png, for pasting into documents
BLUES=['#cde2fb','#b7d3f6','#9ec5f4','#86b6ef','#6da7ec','#5598e7','#3987e5','#2a78d6','#256abf','#1c5cab','#184f95','#104281','#0d366b']
cmap=LinearSegmentedColormap.from_list('fanblue',BLUES)
cmap.set_bad('#d9d8d3')     # no logging
cmap.set_under('#ffffff')   # logged, no fan

# ---- hourly temperature for 10025 (Open-Meteo archive, cached) ----
WX=str(HERE/'wx_archive.json')
if not os.path.exists(WX):
    url=('https://archive-api.open-meteo.com/v1/archive?latitude=40.7963&longitude=-73.9681'
         '&start_date=2026-08-04&end_date=2026-10-06&hourly=temperature_2m'
         '&temperature_unit=fahrenheit&timezone=America%2FNew_York')
    urllib.request.urlretrieve(url,WX)
_wx=json.load(open(WX))['hourly']
TEMP={datetime.datetime.fromisoformat(t):v for t,v in zip(_wx['time'],_wx['temperature_2m'])}

# ---- coverage from logs (minutes logged per date-hour) ----
require_data()
cov={}
for f in log_files():
    for r in csv.DictReader(open(f)):
        try: t=datetime.datetime.fromisoformat(r['time'])
        except Exception: continue
        cov[(t.date(),t.hour)]=cov.get((t.date(),t.hour),0)+1

# ---- events ----
ev=list(csv.DictReader(open(EVENTS_CSV)))
mins={}
for e in ev:
    d=datetime.date.fromisoformat(e['date'])
    sh,sm=map(int,e['start'].split(':')); dur=int(e['duration_min'])
    t=datetime.datetime.combine(d,datetime.time(sh,sm))
    for k in range(dur):
        tt=t+datetime.timedelta(minutes=k)
        mins[(tt.date(),tt.hour)]=mins.get((tt.date(),tt.hour),0)+1

dates=sorted({datetime.date.fromisoformat(e['date']) for e in ev} | {d for d,_ in cov})
d0,d1=min(dates),max(dates)
alld=[d0+datetime.timedelta(days=i) for i in range((d1-d0).days+1)]
M=np.full((24,len(alld)),np.nan)
for j,d in enumerate(alld):
    for h in range(24):
        if cov.get((d,h),0)>=20:            # hour considered logged
            M[h,j]=mins.get((d,h),0)
SEASON=datetime.date(2026,9,17)
act=[d for d in alld if d<=SEASON]
tot_h=np.array([np.nansum([mins.get((d,h),0) for d in act if cov.get((d,h),0)>=20]) for h in range(24)])
hrs_logged=np.array([sum(1 for d in act if cov.get((d,h),0)>=20) for h in range(24)])
rate=np.divide(tot_h,np.maximum(hrs_logged,1))

fig=plt.figure(figsize=(13.5,6.4),dpi=200,facecolor=SURF)
gs=fig.add_gridspec(2,3,width_ratios=[4.6,0.13,1.0],height_ratios=[1,4.4],wspace=0.05,hspace=0.07,
                    left=0.065,right=0.975,top=0.765,bottom=0.20)
axT=fig.add_subplot(gs[0,0]); axT.set_facecolor(SURF)
cax=fig.add_subplot(gs[1,1])
ax=fig.add_subplot(gs[1,0],sharex=axT); ax.set_facecolor(SURF)
bounds=[1,5,10,20,30,45,61]
norm=BoundaryNorm(bounds,cmap.N,extend='min')
im=ax.imshow(M,aspect='auto',origin='lower',cmap=cmap,norm=norm,
             extent=[-0.5,len(alld)-0.5,-0.5,23.5],interpolation='nearest')
ax.set_yticks(range(0,24,3)); ax.set_yticklabels([f'{h:02d}:00' for h in range(0,24,3)],fontsize=8,color=INK2)
ax.set_xticks(np.arange(-0.5,len(alld),1),minor=True); ax.set_yticks(np.arange(-0.5,24,1),minor=True)
ax.grid(which='minor',color=SURF,lw=0.5); ax.set_axisbelow(False)
tick=[j for j,d in enumerate(alld) if d.day in (1,5,10,15,20,25)]
ax.set_xticks(tick); ax.set_xticklabels([alld[j].strftime('%b %-d') for j in tick],fontsize=8,color=INK2)
ax.set_ylabel('Hour of day',fontsize=9,color=INK2)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
# Sept 17 marker
j17=alld.index(datetime.date(2026,9,17))
ax.axvline(j17+0.5,color='#d95926',lw=1.6)
ax.text(j17+0.9,23.0,'last heard\nSep 17',fontsize=8,color='#d95926',va='top',ha='left',linespacing=1.25)

# ---- temperature strip ----
tmax=np.array([max([TEMP[h] for h in TEMP if h.date()==d], default=np.nan) for d in alld])
tmin=np.array([min([TEMP[h] for h in TEMP if h.date()==d], default=np.nan) for d in alld])
xs=np.arange(len(alld))
axT.fill_between(xs,tmin,tmax,color='#eb6834',alpha=0.18,linewidth=0)
axT.plot(xs,tmax,color='#eb6834',lw=1.6)
axT.axhline(75,color=MUTED,lw=0.8,ls=(0,(4,3)))
axT.text(len(alld)-0.5,75,' 75°F',fontsize=7,color=MUTED,va='center',ha='left')
axT.set_ylabel('°F',fontsize=8,color=INK2)
axT.set_ylim(min(np.nanmin(tmin)-3,55),np.nanmax(tmax)+4)
axT.tick_params(labelbottom=False,length=0,labelsize=7,colors=INK2)
axT.set_yticks([60,80])
axT.grid(axis='y',color='#ecebe7',lw=0.6); axT.set_axisbelow(True)
for sp in axT.spines.values(): sp.set_visible(False)
axT.axvline(j17+0.5,color='#d95926',lw=1.6)
axT.set_title('Daily high and low temperature, 10025',fontsize=8,color=MUTED,loc='left',pad=4)

cb=fig.colorbar(im,cax=cax,ticks=[1,5,10,20,30,45,61],extend='min')
cb.ax.set_yticklabels(['1','5','10','20','30','45','60'],fontsize=7,color=INK2)
cb.ax.set_title('min/hr',fontsize=7.5,color=INK2,pad=5)
cb.outline.set_visible(False); cb.ax.tick_params(length=0)

ax2=fig.add_subplot(gs[1,2],sharey=ax); ax2.set_facecolor(SURF)
ax2.barh(range(24),rate,height=0.74,color='#2a78d6')
ax2.set_xlabel('avg minutes per logged hour',fontsize=8,color=INK2)
ax2.tick_params(labelleft=False,length=0,labelsize=7,colors=INK2)
ax2.grid(axis='x',color='#e8e7e3',lw=0.6); ax2.set_axisbelow(True)
for s in ax2.spines.values(): s.set_visible(False)
night=np.mean(rate[list(range(22,24))+list(range(0,7))]); day=np.mean(rate[9:21])
ax2.set_title('Aug 4 – Sep 17',fontsize=8,color=MUTED,pad=6)

fig.text(0.065,0.950,'Rooftop fan activity at 156 W 95th St, detected at 150 W 95th St Apt 6C',
         fontsize=13.5,color=INK,ha='left',weight='bold')
fig.text(0.065,0.895,'282 episodes detected from continuous sound-level logging, 4 Aug – 6 Oct 2026. '
         'White = logged but quiet, grey = no logging. Detection: noise floor (LA90) ≥4 dB above rolling baseline,\n'
         'sharp onset and drop, 5–35 min, and steady (LAeq−LA90 ≤ 2 dB) to exclude speech and radio; 56 of the 67 episodes\ncovered by a saved spectrogram also show the fan’s 1.3–3.5 kHz signature.',
         fontsize=8.3,color=INK2,ha='left',va='top',linespacing=1.6)
fig.text(0.065,0.105,f'Median episode 9 min, noise floor rising 43.1 → 49.3 dB(A) (+6.2 dB); NYC Noise Code §24-227 limit for this equipment is 42 dB(A) indoors.   '
         f'Night (22:00–06:00) averages {night:.1f} min/hour vs {day:.1f} min/hour daytime.\n'
         'Fan activity tracks the heat: on the 18 logged days whose high stayed below 75°F the median was 0 min/hour; above 75°F the median was 2.4–4.1 min/hour. '
         'Temperature: Open-Meteo reanalysis for 40.80°N, 73.97°W.',
         fontsize=8,color=MUTED,ha='left',va='top',linespacing=1.6)
fig.savefig(HERE/'fan_activity.png',facecolor=SURF)
# The document copy, downscaled rather than re-rendered at a lower dpi: the
# same point sizes hint differently at 96 dpi and the footnote reflows off the
# right edge, whereas downscaling a 2x render keeps the layout and supersamples
# the text. The README promised this file but nothing made it, so regenerating
# the chart used to leave it silently stale.
from PIL import Image
big=Image.open(HERE/'fan_activity.png')
big.resize((DOC_WIDTH_PX,round(big.height*DOC_WIDTH_PX/big.width)),
           Image.LANCZOS).save(HERE/'fan_doc.png')
print('ok  night %.2f day %.2f'%(night,day))
print('hour rates:', np.round(rate,1))
