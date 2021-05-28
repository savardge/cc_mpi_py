from obspy.signal.cross_correlation import correlate
from scipy.signal import resample
import numpy as np


def enum(*sequential, **named):
    """
    Way to fake an enumerated type in Python
    Pulled from:  http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)


def upsample(cc, M, df):
    """
    Upsamples a cc array by a rate of M

    Parameters
    ----------
    cc: array_like
        Cross correlation values
    M: int
        Factor by which to upsample

    Returns
    -------
    y: array_like
        Upsampled cross-correlation values
    timing_st_up: array_like
        Timing array centered on upsampled cc values
    max_index_st_up: int
        Index of upsampled cc maximum
    dt_st_up: float
        Timing value of upsampled cc maximum
    """

    y = resample(cc, M * (cc.shape[0]))
    timing_st_up = np.linspace(-1 * int(len(cc) / 2), int(len(cc) / 2) + 1, num=len(y), endpoint=True) / df

    max_index_st_up = np.argmax(y)
    dt_st_up = timing_st_up[max_index_st_up]
    return y, timing_st_up, max_index_st_up, dt_st_up


def find_nearest(array, value):
    """
    Finds the nearest array index to a specific value

    Parameters
    ----------
    array: array_like
        Array to search for index
    value: float
        Value to search array for

    Returns
    -------
    idx : int
        Idx of value in array
    """
    idx = (np.abs(array - value)).argmin()
    return idx


def cross_correlate(tr_1, tr_2, df):
    """
    Cross-correlates two arrays - filtered traces from obspy

    Parameters
    ----------
    tr_1: array_like
        First array to cross-correlate
    tr_2: array_like
        Second array to cross-correlate

    Returns
    -------
    cc: array_like
        Cross-correlation values
    timing_st: array_like
        Timing array centered on cc
    max_index_st: int
        Index of cc maximum
    dt_st: float
        Timing value of cc maximum
    """

    #len1 = len(tr_1)
    cc = correlate(tr_1, tr_2, shift=int((len(tr_1)) / 2))
    timing_st = np.linspace(-1 * int(len(tr_2) / 2), int(len(tr_2) / 2), num=len(cc), endpoint=True) / df

    max_index_st = np.argmax(cc)
    dt_st = timing_st[max_index_st]
    return cc, timing_st, max_index_st, dt_st
