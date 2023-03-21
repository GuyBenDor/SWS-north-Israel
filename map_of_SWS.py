import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import haversine, set_Mfigure, add_rose

area = "all_stations"
# areas = ["korazim"]
min_rec, min_corr_coef = 0.75, 0.75
min_snr0 = 0
min_snrf = 4.5
min_num = 10
max_var = 40
max_std = 90
max_dist = 26
confidence_level = 0

plot_specs = dict(
    st_order=[
        ["MGDL","SPR","KNHM","AMID","RSPN",'ZEFT',"DSHN","KRSH"],
        ["MSAM","MTLA","GOSH","GEM","NATI"],
        ["LVBS","HULT","KDZV","KSH0","HDNS","RMOT","ENGV"],
        ["GLHS",'MNHM',"ALMT",'TVR']
    ],
    start_row=[0.05, 0.14, 0.1, 0.27],
    ds=[0.13, 0.2, 0.13, 0.2]
)


# p_param = dict(incidence=90, snr=1)

drop_confidence = True

# fig_type = "rose"    # either "rose" or "arrow"

arrow_dict = {
    'scale': 18., 'width': 0.005, 'headwidth': 1.0, 'headlength': .1,'alpha':0.9# "scale_units": "inches",
}

region = [35.3, 35.9, 32.5, 33.5]
view_bounds = [0.1, -0.0, 0.1, -0.15]


event_path = "events.csv"
# event_path = "../data/SWS_Guy_Picks_assoc_DB_Origins.txt"
site_path = "sites.csv"
split_path = "SWS_CC.csv"
pick_path = "sWavePicks.csv"
filepath = "/Users/guy.bendor/PycharmProjects/seisMaps/data4paper1/all_stations.xlsx"


df = pd.read_csv(split_path,dtype={"evid": str})

sites = pd.read_csv(site_path)

df = df.merge(sites, on="station", how="left")

picks = pd.read_csv(pick_path, dtype={'evid': str})
df = df.merge(picks, on=["evid", "station"], how="left")

events = pd.read_csv(event_path,dtype={'evid':str})
events = events[["evid", 'lat', 'lon', 'depth', 'UTCtime', 'ms']]
df = df.merge(events, on='evid',how='left')
df['UTCtime'] = np.array(df['UTCtime'].str[:-1].values, dtype="datetime64[us]")
# print(df.pickTime)
df['pickTime'] = np.array(df['pickTime'].values, dtype="datetime64[us]")

# df.sort_values(by=["station", "evid"], inplace=True, ignore_index=True)



stations = list(set(list(itertools.chain(*plot_specs['st_order']))))
df = df[df.station.isin(stations)]

# print(df.columns)


df["region"] = np.nan

# df.loc[
#     (df.lat >= region[2]) & (df.lat <= region[3]) &
#     (df.lon >= region[0]) & (df.lon <= region[1]),
#     "region"
# ] = area

# df.dropna(subset=["region"], inplace=True)
df = df.merge(df.groupby("station").size().rename("total_num"), left_on="station",right_index=True,how="left")

df['epi_dist'] = df[["longitude", "latitude", "lon", 'lat']].apply(lambda x: haversine(*x), axis=1)

# df = df[
#     (df.confidence <= confidence_level) &
#     (df.snr_s >= min_snr) &
#     (df.rec_s >= min_rec) &
#     (df.epi_dist<=max_dist)
#     & (df.cc >= min_corr_coef)
# ]
#
# if 'snr_f' in df.columns:
#     df = df[df.snr_s>=min_snr]
#
#
# df['angle_s'] = df['angle_s']%360

df = df[
    (df.snr0 >= min_snr0) &
    (df.snrf >= min_snrf) &
    # (df.ang_diff <= min_dphi) &
    (df.Null!=True) &
    (df.rec>=min_rec) &
    (df.cc>=min_corr_coef) &
    (df.epi_dist <= max_dist)
]

# print(len(df))
df['angle_s'] = df['ang']



dat = df[["station", "epi_dist"]].groupby("station").agg(np.max)


dat = dat.merge(df[["station", "evid"]].groupby("station").agg(len), left_index=True, right_index=True)

rect = [round(s+b,3) for s,b in zip(region, view_bounds)]


fig, ax = set_Mfigure(rect,df=df)


ax = add_rose(fig, ax, df, plot_specs,filepath, min_num=min_num,max_var=max_var, max_std=max_std,arrows=arrow_dict)
import figure_codes.scaler_bar as sb
sb.scale_bar(ax, 10,location=[0.8,0.07])


# import cartopy.crs as ccrs
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# import matplotlib.ticker as mticker
# gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.5, color='black', alpha=0.3, draw_labels=True)
# gl.top_labels = True
# gl.left_labels = True
# gl.right_labels = True
# gl.xlines = False
# gl.ylines = False
# gl.xlocator = mticker.FixedLocator(np.arange(32, 37, 0.2))
# gl.ylocator = mticker.FixedLocator(np.arange(29,35, 0.2))
# gl.xformatter = LONGITUDE_FORMATTER
# gl.yformatter = LATITUDE_FORMATTER
#
# font1 = {'family': 'serif',
#         'color':  'darkblue',
#         'weight': 'bold',
#         'size': 10,
#         }
# ax.text(35.61,32.9,'JGF',font1,rotation=85)

# fig.savefig(f'/Users/guy.bendor/Documents/paper 1/figures/fig {area}.png', bbox_inches='tight', pad_inches=0.1,dpi=700)
plt.show()
