import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from scipy.stats import circstd, circmean, circvar
import math
from utils import haversine, initial_bearing, rosePlot
from matplotlib.gridspec import GridSpec



min_snr0 = 0
min_snrf = 4.5
min_rec = 0.75
min_cc = 0.75
min_dphi = 90
Types = ['FSR', 'CC']
conf = 0
max_dist = 26

def ang_diff(v1_u, v2_u):
    return np.degrees(np.arccos(np.clip(v2_u@v1_u, -1.0, 1.0)))
df = pd.read_csv('SWS_CC.csv',dtype={'evid':str})
# df['y'] = np.cos(np.radians(df.ang))
# df['x'] = np.sin(np.radians(df.ang))

events = pd.read_csv(f'events.csv', dtype={"evid": str})
picks = pd.read_csv(f'sWavePicks.csv', dtype={"evid": str})
# picks = picks[picks.Phase == 'S']
sites = pd.read_csv(f'sites.csv')
df = df.merge(events, on='evid', how='left')
df = df.merge(picks, on=['station', 'evid'], how='left')
df = df.merge(sites, on='station', how='left')

df['pickTime'] = np.array(df['pickTime'].values,dtype='datetime64[us]')
df['dist'] = df[['lon','lat','longitude','latitude']].apply(lambda x: haversine(*x),axis=1)
df['hypo'] = df[['dist','depth']].apply(lambda x: np.sqrt(x[0]**2+x[1]**2),axis=1)
df["epi_dir"] = df[['lon','lat','longitude','latitude']].apply(lambda x: initial_bearing(*x),axis=1)
print(len(df))
df = df[
    (df.snr0 >= min_snr0) &
    (df.Null!=True) &
    # (df.confidence<=conf) &
    (df.snrf >= min_snrf) &
    (df.cc >= min_cc) &
    (df.rec >= min_rec) &
    (df.dist <= max_dist)# & (df.station.isin(['GLHS',"KDZV"]))
]
print(len(df))

df['rad'] = np.radians(df.ang)
strikeDat = df[['station', "rad"]].groupby("station").agg(
    lambda x: [
        circmean(x, low=-np.pi/2, high=np.pi/2),
        circstd(x, low=-np.pi/2, high=np.pi/2),
        circvar(x, low=-np.pi/2, high=np.pi/2),
        len(x)
    ]
)

strikeDat[["meanRad", "stdRad", "varRad",'gsize']] = np.array(strikeDat.rad.tolist())
strikeDat.drop(columns="rad", inplace=True)
strikeDat[['meanAngle', 'Var', 'Std']] = np.round(np.degrees(strikeDat[["meanRad",  "varRad","stdRad"]]),0)
print(strikeDat[['meanAngle', 'Var', 'Std','gsize']].sort_values(by='Var',ascending=False))
print()

binwidth = 0.01
bins = np.arange(-0.005,0.505 + binwidth, binwidth)
fig,ax =plt.subplots()
ax.hist(
                df.dt.values,
                bins, edgecolor='black', linewidth=1.2, alpha=0.75
            )
plt.show()




for sta in df.station.unique():
    stats = strikeDat[strikeDat.index==sta].iloc[0]

    temp = df[df.station == sta].copy()
    print(sta, np.mean(temp.dt.values))
    fig = plt.figure(figsize=[16, 9])
    gs0 = GridSpec(1, 1, left=0.1, right=0.4, wspace=0.1, hspace=0.1)
    gs1 = GridSpec(2, 2, left=0.45, right=0.95, wspace=0.1, hspace=0.4,bottom=0.1,top=0.7)
    ax0 = [fig.add_subplot(gs0[0], polar=True)]
    ax0 += [fig.add_subplot(gs1[0,j]) for j in range(2)]
    ax0 += [fig.add_subplot(gs1[1, j]) for j in range(2)]

    rosePlot(ax0[0], temp.ang.values)
    ax0[1].hist(
                temp.dt.values,
                bins, edgecolor='black', linewidth=1.2, alpha=0.75
            )
    ax0[1].set_title(r'Histogram of $\delta$t')
    ax0[2].scatter(temp.hypo, temp.dt,ec='k',lw=0.5,s=30)
    ax0[2].set_title(r'$\delta$t as a function of hypo')

    ax0[3].scatter(temp.epi_dir, temp.dt,ec='k',lw=0.5,s=30)
    ax0[3].set_title(r'$\delta$t as a function of epi direction')
    ax0[4].scatter(temp.depth, temp.dt,ec='k',lw=0.5,s=30)
    ax0[4].set_title(r'$\delta$t as a function of depth')
    # g = temp.groupby(pd.Grouper(key="pickTime", freq="M")).agg({"dt": ["mean", "median", "std", "count"]})
    # g.columns = g.columns.droplevel()
    # g = g[g["count"] > 0]
    # gg = g.dropna().copy()
    # gg['date'] = gg.index - np.timedelta64(15, 'D')
    # ax0[2].errorbar(gg['date'], gg["mean"], yerr=gg["std"], label='monthly avg', c="tab:orange", zorder=0)
    tx1 = f'mean = {stats.meanAngle}\nvar = {stats.Var}\nstd = {stats.Std}\nsize = {int(stats.gsize)}'
    plt.text(0.42, 0.8, tx1, fontsize=11, transform=plt.gcf().transFigure, ha='left', style='italic')




    # gs1 = GridSpec(2, 1, left=0.5, right=0.98, wspace=0.0, hspace=0.12)
    # ax1 = [fig.add_subplot(gs1[j], polar=True) for j in range(2)]
    # yy = [0.8,0.4]
    # for i in range(len(Types)):
    #     tt = Types[i]
    #     cols = [col for col in temp.columns if tt in col]
    #     cols += ['station', 'evid', 'tw', 'snr0', 'Null', 'chan']
    #     temp1 = temp[cols].copy()
    #     temp1.columns = temp1.columns.str.rstrip(f'_{tt}')
    #     # temp1['snr_ratio'] = np.where(temp1.snrf<temp1.snr0,False,True)
    #     temp1 = temp1[
    #         # (temp1.snrf>=min_snrf) &
    #         (temp1.cc >=min_cc) & #(temp1.snr_ratio==True) &
    #         (temp1.rec >= min_rec)
    #     ]
    #     ax0[i].hist(
    #         temp1.td.values,
    #         bins, edgecolor='black', linewidth=1.2, alpha=0.75
    #     )
    #     rosePlot(ax1[i],temp1.ang.values)
    #     # temp1.loc[temp1.ang>90,'ang'] = temp1.loc[temp1.ang>90,'ang']+180
    #     temp1['rad'] = np.radians(temp1.ang)
    #     Var = round(np.degrees(circvar(temp1.rad.values, low=-np.pi/2, high=np.pi/2)),1)
    #     Mean = round(np.degrees(circmean(temp1.rad.values, low=-np.pi/2, high=np.pi/2)),1)
    #     Std = round(np.degrees(circstd(temp1.rad.values, low=-np.pi/2, high=np.pi/2)),1)
    #     tx1 = f'mean = {Mean}\nvar = {Var}\nstd = {Std}'
    #     tx2 = f'{tt} Method'
    #     plt.text(0.42, yy[i], tx1, fontsize=11, transform=plt.gcf().transFigure, ha='left', style='italic')
    #     plt.text(
    #         0.07, yy[i] - 0.1, tx2, fontsize=14, transform=plt.gcf().transFigure, ha='left', style='italic',
    #         rotation=90, weight='extra bold'
    #     )
    fig.suptitle(sta)
    plt.show()

# ax[0].hist(
#     df[(df.rec_FSR>=0.7)]['td_FSR'],
#     bins, edgecolor='black', linewidth=1.2, alpha=0.75
# )
#
# ax[1].hist(
#     df[(df.cc_CC>=0.7)]['td_CC'],
#     bins, edgecolor='black', linewidth=1.2, alpha=0.75
# )



# df = df[df.ang_diff < 30]