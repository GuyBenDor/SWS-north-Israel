import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from supplemental_functions import haversine, streamData, creating_fig, sliding_window
import os
from obspy.signal.cross_correlation import correlate_template
from scipy.signal import periodogram


pick_path = f"sWavePicks.csv"
# pick_path = "../pickingEvents/pickData.csv"
event_path = 'events.csv'
# event_path = '../data/relocated3.csv'
station_path = 'sites.csv'
stream_path = '/Users/guy.bendor/Traces/StreamForPca'
save_path = 'SWS_CC'

smp_slice = 700
corr_tw = 0.7
corr_small_tw = 0.3
noise_tw = 1
noise_in_sig = 0.4
ind_skip = 0
init_tw = 0.05
snr_portion = 0.7
plot_figs = True
plot_figs = False

save_df = True
# save_df = False

plotType = 'cc'
twRange = np.arange(0.15, 0.41, 0.01)
twSize_criterion = 0.7
cc_percent = 1  # .99
event = 0#276
# 61#41 #70
# print(len(twRange))


filters = {
    "HH": [2, 12],
    "EN": [1, 12]
}

stData = streamData(smp_slice)
start_noise = smp_slice - int(noise_tw * 200)
start = smp_slice - int(noise_in_sig * 200)
angs = np.arange(-90, 90)
dts = np.arange(0, corr_tw, 0.005)
X, Y = np.meshgrid(dts, angs)

c, s = np.cos(np.radians(angs)), np.sin(np.radians(angs))
M = np.zeros((2, 2, len(angs)))
M[0, 0, :] = c
M[0, 1, :] = -s
M[1, 0, :] = s
M[1, 1, :] = c

trnames = [['E-W', 'N-S'], ['slow', 'fast'], ['slow-shifted', 'fast']]

reorientation = {
    # KDZV-EN station was re-oriented by 68 deg clockwise at 2021-02-15
    'KDZV': {
        'chan': 'EN',
        'fixDate': np.datetime64('2021-02-15 13:50:00'),
        'fixAngle': -68
    },
    # GLHS-HH station was re-oriented by 115 deg counterclockwise at 2018-07-10
    'GLHS': {
        'chan': 'HH',
        'fixDate': np.datetime64('2018-07-11 00:00:00'),
        'fixAngle': -115  # 245
    }
}

picks = pd.read_csv(pick_path, dtype={"evid": str})
picks['pickTime'] = pd.to_datetime(picks['pickTime'])

events = pd.read_csv(event_path, dtype={'evid': str})
picks = picks.merge(events, on='evid', how='left')
sites = pd.read_csv(station_path)
picks = picks.merge(sites, on='station', how='left')
picks['dist'] = picks[['longitude', 'latitude', 'lon', 'lat']].apply(lambda xx: haversine(*xx), axis=1)

# picks = picks[
#     (picks.Phase=="S") &
#     # (picks.confidence==0) &
#     (picks.dist<=26)
# ]
# picks = picks[
#     (picks.station.isin(['KNHM']))  # &
#     & (picks.evid.isin(['125814']))  # & (picks.station.isin(['RSPN']))
#     ]
# 122277 - KNHM
# 122281 - KNHM
# 122282 - KNHM
# 122302 - KNHM
# 122372 - KNHM
# 122413 - KNHM
# 122461
picks.reset_index(inplace=True, drop=True)

print('Reading event miniseeds...')
inds = []
chans = []
data = []
for num in range(event, len(picks)):  # event

    row = picks.iloc[num]
    if num % 1 == 0:
        print(f'\titer #{num} of {len(picks)}\t\t{row.evid}')
    trace_name = f'{row.evid}.mseed'
    path = None
    for key, val in filters.items():
        pathtmp = os.path.join(stream_path, row.station, key, trace_name)
        if os.path.exists(pathtmp):
            f_val = val
            path = pathtmp
            chan = key
            break

    if path is None:
        continue

    Null = False

    arr = stData(path, *f_val, row.pickTime)[:2]

    if row.station in reorientation.keys():
        if row.pickTime < reorientation[row.station]['fixDate'] and chan == reorientation[row.station]['chan']:
            fa = np.radians(reorientation[row.station]['fixAngle'])
            co, so = np.cos(fa), np.sin(fa)
            Rorient = np.array([[co, so], [-so, co]])
            arr = Rorient @ arr

    for tw in twRange:
        end1 = smp_slice + int(tw * 200)
        end2 = end1 + int(corr_tw * 200)
        stmp = np.linalg.svd(
            arr[:, smp_slice:end1] - arr[:, smp_slice:end1].mean(axis=-1)[..., np.newaxis], compute_uv=False
        )
        rtmp = 1 - stmp[1] / stmp[0]
        if rtmp <= twSize_criterion:
            break

    if rtmp > twSize_criterion:
        Null = True

    arrR = np.matmul(M.T, arr)

    RecMat = np.zeros((len(angs), len(dts)))
    CCMat = np.zeros((len(angs), len(dts)))
    snrMat = np.zeros((len(angs), len(dts)))
    Mtmps = np.zeros((len(angs), len(dts), 2, 2))
    # tws = np.zeros(len(angs))

    for a in range(len(angs)):

        ArrSingnal = np.zeros([len(dts), 2, end1 - start])
        ArrSingnal[:, 1] = arrR[a, 1, start:end1][np.newaxis]
        ArrSingnal[:, 0] = sliding_window(arrR[a, 0, start:end2 - 1], end1 - start)

        ArrNoise = np.zeros([len(dts), 2, smp_slice - start_noise])
        ArrNoise[:, 1] = arrR[a, 1, start_noise:smp_slice][np.newaxis]
        ArrNoise[:, 0] = sliding_window(arrR[a, 0, start_noise:smp_slice + (end2 - end1) - 1], smp_slice - start_noise)

        rmsS = np.sqrt((ArrSingnal[..., smp_slice - start:] ** 2).sum(-2).mean(-1))
        rmsN = np.sqrt((ArrNoise ** 2).sum(-2).mean(-1))
        snrMat[a, :] = rmsS / rmsN
        del ArrNoise, rmsS, rmsN

        if plot_figs:
            U, s1, V = np.linalg.svd(
                ArrSingnal - ArrSingnal[..., :smp_slice - start].mean(axis=-1)[..., np.newaxis]
            )
            rec = 1 - s1[:, 1] / s1[:, 0]
            RecMat[a, :] = rec
            angtmp = np.degrees(np.arctan2(U[:, 0, 0], U[:, 0, 1])) % 360
            del U, s1, V, rec

            ctmp, stmp = np.cos(np.radians(angtmp)), np.sin(np.radians(angtmp))
            Mtmp = np.zeros((2, 2, len(angtmp)))
            Mtmp[0, 0, :] = ctmp
            Mtmp[0, 1, :] = -stmp
            Mtmp[1, 0, :] = stmp
            Mtmp[1, 1, :] = ctmp
            Mtmps[a, ...] = np.linalg.inv(Mtmp.T)
        else:
            s1 = np.linalg.svd(
                ArrSingnal - ArrSingnal[..., :smp_slice - start].mean(axis=-1)[..., np.newaxis], compute_uv=False
            )
            rec = 1 - s1[:, 1] / s1[:, 0]
            RecMat[a, :] = rec
        CCMat[a, :] = correlate_template(arrR[a, 0, start:end2 - 1], arrR[a, 1, start:end1])

    snrMat = np.round(snrMat, 2)
    CCMat = np.abs(CCMat)
    snr0 = snrMat[:, 0].min()
    ccrlMat = np.multiply(snrMat / snr0, CCMat)

    try:
        inx_cc = np.argwhere(CCMat == CCMat[snrMat > snr0].max())[0]
    except:
        # print(f'\titer #{num} of {len(picks)}\t\t{row.evid}')
        # print('Issue')
        continue

    inx_rec = np.argwhere(RecMat == RecMat.max())[0]
    inx_snr = np.argwhere(snrMat == snrMat.max())[0]
    inx_ccrl = np.argwhere(CCMat == CCMat.max())[0]
    inx_bs = inx_cc.copy()

    ang, dt = angs[inx_bs[0]], dts[inx_bs[1]]
    snr_final = snrMat[inx_bs[0], inx_bs[1]]
    rec = RecMat[inx_bs[0], inx_bs[1]]
    cc = CCMat[inx_bs[0], inx_bs[1]]
    # tw = tws[inx_bs[0]]


    if plot_figs:
        print(f'inx: {num}\nsta: {row.station}\nevid: {row.evid}\n\tang: {-ang}\n\tdt: {dt}\n\trec: {round(rec, 2)}'
              f'\n\tsnr: {round(snr0, 2)}\n\tsnr final: {round(snr_final, 2)}\n\tcc: {round(cc, 2)}')
        print("--------")
        CCMat = np.flip(CCMat, 0)
        RecMat = np.flip(RecMat, 0)
        ccrlMat = np.flip(ccrlMat, 0)
        snrMat = np.flip(snrMat / snr0, 0)

        rot = np.linalg.inv(M[:, :, inx_bs[0]]) @ arr
        corr = np.zeros([2, arr.shape[1] - inx_bs[1]])
        corr[0] = rot[0, inx_bs[1]:]
        if not inx_bs[1] == 0:
            corr[1] = rot[1, :-inx_bs[1]]
        else:
            corr[1] = rot[1]
        plotArrs = [arr, rot, corr]

        line = np.zeros([2, 2])
        line[:, 0] = [-s[inx_bs[0]], c[inx_bs[0]]]
        line[:, 1] = -line[:, 0]
        line1 = np.linalg.inv(M[:, :, inx_bs[0]]) @ line
        line2 = Mtmps[inx_bs[0], inx_bs[1]].T @ line1
        lines = [line, line1, line2]

        fig, ax0, ax1, ax2, ax3 = creating_fig()

        for i, temp in enumerate(plotArrs):

            # print(temp.shape)
            ax0[i].plot(
                *(temp[:, start:end1] / np.linalg.norm(temp[:, start:end1], axis=0).max()), color='k', lw=0.7, zorder=1
            )
            color = 'green' if i != 2 else 'red'
            ax0[i].plot(*lines[i], color=color, zorder=0, lw=0.7)

            ax0[i].set_xlim(-1, 1)
            ax0[i].set_ylim(-1, 1)
            ax0[i].set_xticks([])
            ax0[i].set_yticks([])
            ax0[i].set_aspect('equal', 'box')

            x = np.arange(temp.shape[1]) / 200
            for j in range(2):
                ax1[i][j].plot(x, temp[j], color='k', lw=0.7)
                ax1[i][j].axvline(x=start / 200, color='k', ls='--', lw=0.5)
                ax1[i][j].axvline(x=end1 / 200, color='k', ls='--', lw=0.5)
                ax1[i][j].text(0.05, 0.8, trnames[i][j], style='italic', fontsize=8,
                               bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 3},
                               transform=ax1[i][j].transAxes)

                if i == 1 and j == 0:
                    ax1[i][j].axvline(x=(start + inx_bs[1]) / 200, color='k', lw=0.3)
                    ax1[i][j].axvline(x=(end1 + inx_bs[1]) / 200, color='k', lw=0.3)
                    ax1[i][j].axvspan((start + inx_bs[1]) / 200, (end1 + inx_bs[1]) / 200, alpha=0.1, color='green')
                    ax1[i][j].axvline(x=(smp_slice + inx_bs[1]) / 200, color='r')
                else:
                    ax1[i][j].axvline(x=smp_slice / 200, color='r')
                    if not i == 0:
                        ax1[i][j].axvspan(start / 200, end1 / 200, alpha=0.1, color='green', zorder=1)
                ax1[i][j].set_yticks([])
                if not (i == 2 and j == 0):
                    plt.setp(ax1[i][j].get_xticklabels(), visible=False)
                else:
                    ax1[i][j].set_xlabel('Time [sec]')
                    ax1[i][j].set_xlim(0, x[-1])


        extent = dts.min(), dts.max(), angs.min(), angs.max()
        # FSratioMat = np.flip(FSratioMat,0)
        inx_bs[0] = -inx_bs[0]  # %180
        inx_rec[0] = -inx_rec[0]
        inx_cc[0] = -inx_cc[0]  # % 180
        inx_snr[0] = -inx_snr[0]  # % 180
        inx_ccrl[0] = -inx_ccrl[0]

        im = ax2[0].imshow(RecMat, origin='lower', extent=extent, aspect="auto", alpha=0.7, cmap="plasma",vmin=0,vmax=1)
        CS = ax2[0].contour(X, Y, RecMat, np.arange(0.1, 1, 0.1), alpha=0.4, colors='k')
        ax2[0].clabel(CS, inline=True, fontsize=7)
        ax2[0].scatter(dts[inx_rec[1]], angs[inx_rec[0]], s=40, zorder=10, c="tab:blue", ec='k', lw=0.5, label='rl max')
        ax2[0].scatter(dts[inx_bs[1]], angs[inx_bs[0]], s=60, zorder=11, marker="x", c="k", ec='k', lw=0.5)
        ax2[0].set_title('Rectilinearity', fontsize=10, y=0.98)
        plt.setp(ax2[0].get_xticklabels(), visible=False)

        im = ax2[1].imshow(CCMat, origin='lower', extent=extent, aspect="auto", alpha=0.7, cmap="plasma",vmin=0,vmax=1)
        CS = ax2[1].contour(X, Y, CCMat, np.arange(0.1, 1, 0.1), alpha=0.4, colors='k')
        ax2[1].clabel(CS, inline=True, fontsize=7)
        ax2[1].scatter(dts[inx_ccrl[1]], angs[inx_ccrl[0]], s=40, zorder=10, c="tab:red", ec='k', lw=0.5, label='cc max')
        ax2[1].scatter(dts[inx_bs[1]], angs[inx_bs[0]], s=60, zorder=11, marker="x", c="k", ec='k', lw=0.5)
        ax2[1].set_title('Cross Correlation', fontsize=10, y=0.98)
        plt.setp(ax2[1].get_xticklabels(), visible=False)

        im = ax2[2].imshow(snrMat, origin='lower', extent=extent, aspect="auto", alpha=0.7, cmap="plasma")
        CS = ax2[2].contour(X, Y, snrMat, np.arange(0.1, 1.2, 0.1), alpha=0.4, colors='k')
        ax2[2].clabel(CS, inline=True, fontsize=7)
        ax2[2].scatter(dts[inx_snr[1]], angs[inx_snr[0]], s=40, zorder=10, c="tab:green", ec='k', lw=0.5,
                       label='snr max')
        ax2[2].scatter(dts[inx_bs[1]], angs[inx_bs[0]], s=60, zorder=11, marker="x", c="k", ec='k', lw=0.5,
                       label='best-fit')
        ax2[2].set_title('snr/snr init', fontsize=10, y=0.98)
        ax2[2].set_xlabel(r'$\delta$t [s]')
        for i in range(3):
            ax2[i].set_ylabel(r'$\phi^{\circ}$')
            ax2[i].yaxis.set_label_coords(-.08, .5)

        handles, labels = [(a + b + c) for a, b, c in zip(*[ax2[i].get_legend_handles_labels() for i in range(3)])]
        fig.legend(handles, labels, loc=[0.82, 0.76])

        specs = []
        for temp,label in zip(arr[:, start_noise:end1],['EW - component','NS - component']):
            nfft = int(2 ** (np.ceil(np.log2(len(temp)))))  # ensure nfft is power of 2
            freq, spec = periodogram(temp, 200, window="hann", nfft=nfft,
                                     detrend=False, return_onesided=True,
                                     scaling="spectrum")
            ax3[0].loglog(freq, spec,label=label)
        ax3[0].set_ylabel("Power spectrum")
        ax3[0].set_xlabel("Frequency")
        ax3[0].legend()

        if ang > 0:
            tx2 = r'N' + f'{-int(round(-ang, 0))}' + r'$^{\circ}$W'

        else:
            tx2 = r'N' + f'{int(round(-ang, 0))}' + r'$^{\circ}$E'
        tx3 = r'N' + f'{int(round(-ang + 90, 0))}' + r'$^{\circ}$E'
        tx4 = tx3 + f'\nAdvanced by {int(dt * 1000)} ms'
        plt.text(0.05, 0.378, tx3, fontsize=9, transform=plt.gcf().transFigure, ha='left', style='italic')
        plt.text(0.05, 0.5, tx2, fontsize=9, transform=plt.gcf().transFigure, ha='left', style='italic')
        plt.text(0.05, 0.118, tx4, fontsize=9, transform=plt.gcf().transFigure, ha='left', style='italic')
        plt.text(0.05, 0.234, tx2, fontsize=9, transform=plt.gcf().transFigure, ha='left', style='italic')

        tx1 = r'$\phi$ = ' + f'{int(round(-ang, 0))}' + r'$^{\circ}$' + \
              '\n' + r'$\delta$t = ' + f'{int(dt * 1000)} ms\nEpi dist = {round(row.dist, 1)} km'
        tx2 = f'TW = {(end1 - smp_slice) / 200}\nRL = {round(rec, 2)}\nCC = {round(cc, 2)}'
        tx3 = f'SNR initial = {round(snr0, 2)}\nSNR final  =  {round(snr_final, 2)}'
        tx4 = f'Criteria: {plotType}\nNull: {Null}'
        plt.text(0.05, .89, tx1, fontsize=9, transform=plt.gcf().transFigure, ha='left', style='italic')
        plt.text(0.15, .89, tx2, fontsize=9, transform=plt.gcf().transFigure, ha='left', style='italic')
        plt.text(0.25, .89, tx3, fontsize=9, transform=plt.gcf().transFigure, ha='left', style='italic')
        plt.text(0.45, .89, tx4, fontsize=11, transform=plt.gcf().transFigure, ha='left', style='italic')
        fig.suptitle(f'{row.station} - {chan} - {row.evid}')
        plt.show()

    inds.append(num)
    data.append([row.station, chan, row.evid, tw, snr0, Null, -ang, dt, snr_final, rec, cc])

picks = picks.iloc[inds]
picks.reset_index(inplace=True, drop=True)
# data = np.array(data)

colNames = ['station', 'chan', 'evid', 'tw', 'snr0', 'Null', 'ang', 'dt', 'snrf', 'rec', 'cc']

df = pd.DataFrame(data, columns=colNames)
for col in df.columns:
    if col not in ['station', 'evid', 'chan', 'Null']:
        df[col] = df[col].astype(float)
df['ang'] = (df['ang'] + 180) % 180

if save_df:
    df.to_csv(f'{save_path}.csv', index=False)
