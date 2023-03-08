import numpy as np
import obspy
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def find_nearest(array, value):
    # array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def streamData(sample_slice):
    def createEvent(path,fmin,fmax,pickTime):
        st = obspy.read(path)
        st.filter(
            "bandpass", freqmin=fmin, freqmax=fmax, corners=4#, zerophase=True
        )
        st.taper(max_percentage=0.1, type='hann')

        tp = np.array([[tr.stats.starttime, tr.stats.endtime] for tr in st]).T
        st.trim(tp[0].max(), tp[1].min())
        dtimes = np.arange(0, len(st[0].data)) * st[0].stats.delta * 1000
        dtimes = dtimes.astype("timedelta64[ms]") + np.datetime64(str(st[0].stats.starttime)[:-1])
        arr = [st.select(component=t)[0] for t in "ENZ"]
        arr.append(dtimes.astype(float))
        arr = np.array(arr)
        iSt = find_nearest(arr[-1].astype('datetime64[us]'),np.datetime64(pickTime))
        return arr[:, iSt-sample_slice:iSt+sample_slice+1]
    return createEvent

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    :param x: an array containing [lon1,lat1,lon2,lat2]
    :return: a scaler distance in km
    """
    # convert decimal degrees to radians
    lon1, lat1, lon, lat = map(math.radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon - lon1
    dlat = lat - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    km = 6371 * c
    return km

def creating_fig():
    fig = plt.figure(figsize=[16, 9])

    gs0 = GridSpec(3, 1, left=0.26, right=0.7, wspace=0, hspace=0.1)
    ax0 = [fig.add_subplot(gs0[j]) for j in range(3)]

    gs1 = GridSpec(3, 1, left=0.02, right=0.4, wspace=0.0, hspace=0.1)
    ax1 = []
    for j in range(3):
        gs10 = gs1[j].subgridspec(2, 1, hspace=0.)
        if len(ax1) == 0:
            ax_temp = [fig.add_subplot(gs10[1])]
            ax_temp += [fig.add_subplot(gs10[0], sharey=ax_temp[0], sharex=ax_temp[0])]
        else:
            ax_temp = [fig.add_subplot(gs10[i], sharey=ax1[0][0], sharex=ax1[0][0]) for i in range(1, -1, -1)]
        ax1.append(ax_temp)

    gs2 = GridSpec(3, 1, left=0.6, right=0.85, wspace=0.0, hspace=0.08)
    ax2 = [fig.add_subplot(gs2[0])]
    ax2 += [fig.add_subplot(gs2[i], sharey=ax2[0], sharex=ax2[0]) for i in [1,2]]

    gs3 = GridSpec(3, 1, left=0.87, right=0.99, wspace=0.0, hspace=0.08)
    # ax3 = [fig.add_subplot(gs3[2], sharey=ax2[0])]

    labels = [['E-W','N-S'],['Slow','Fast'],['Slow-shifted','Fast']]
    x_coor = [0.5,0.4,0.5]
    for i in range(3):
        ax0[i].set_xlabel(labels[i][0], size=8)
        ax0[i].set_ylabel(labels[i][1], size=8)
        ax0[i].xaxis.set_label_coords(x_coor[i], 0.05)
        ax0[i].yaxis.set_label_coords(-0.005, 0.5)
    return fig, ax0, ax1, ax2#, ax3

def sliding_window(data, size, stepsize=1, padded=False, axis=-1, copy=True):
    """
    Calculate a sliding window over a signal
    Parameters
    ----------
    data : numpy array
        The array to be slided over.
    size : int
        The sliding window size
    stepsize : int
        The sliding window stepsize. Defaults to 1.
    axis : int
        The axis to slide over. Defaults to the last axis.
    copy : bool
        Return strided array as copy to avoid sideffects when manipulating the
        output array.
    Returns
    -------
    data : numpy array
        A matrix where row in last dimension consists of one instance
        of the sliding window.
    Notes
    -----
    - Be wary of setting `copy` to `False` as undesired sideffects with the
      output values may occurr.
    Examples
    --------
    a = numpy.array([1, 2, 3, 4, 5])
    sliding_window(a, size=3)
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    sliding_window(a, size=3, stepsize=2)
    array([[1, 2, 3],
           [3, 4, 5]])
    See Also
    --------
    pieces : Calculate number of pieces available by sliding
    """
    if axis >= data.ndim:
        raise ValueError(
            "Axis value out of range"
        )

    if stepsize < 1:
        raise ValueError(
            "Stepsize may not be zero or negative"
        )

    if size > data.shape[axis]:
        raise ValueError(
            "Sliding window size may not exceed size of selected axis"
        )

    shape = list(data.shape)
    shape[axis] = np.floor(data.shape[axis] / stepsize - size / stepsize + 1).astype(int)
    shape.append(size)

    strides = list(data.strides)
    strides[axis] *= stepsize
    strides.append(data.strides[axis])

    strided = np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides
    )

    if copy:
        return strided.copy()
    else:
        return strided