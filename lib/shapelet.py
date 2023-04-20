import numpy as np
from lib.config import *
from scipy.spatial.distance import cosine

def shapelet_representation(vector, m_0=1000):
    def _similarity_non_flat(vector1, vector2):
        if np.std(vector1) < 1e-100 or np.std(vector2) < 1e-100:
            similarity_value = 0
        else:
            similarity_value = np.corrcoef(vector1, vector2)[0][1]
        return similarity_value
    if m_0 < 0.0005:
        m_0 = 0.0005
    coords = []
    # The average absolute slope of slope_thres
    # gets a flatness of 0.1. Modify below to change
    beta = -np.log(0.1) / m_0

    # flat threshold is m0. If the slope is below m0 flatness is 1
    m0 = 0
    slope = np.mean(abs(np.diff(vector)))
    if slope < m0:
        flatness = 1
    else:
        flatness = np.exp(-beta * (slope - m0));
    for i in range(len(shapelet_array)):
        if not (any(shapelet_array[i])):
            score = 2 * flatness - 1
        else:
            score = (1 - flatness) * _similarity_non_flat(shapelet_array[i], vector)
        coords.append(score)
    return coords

def SS_dist(ts1,ts2,m_01=None,m_02=None):
    shapelet_ts1 = shapelet_representation(ts1,m_0=Thresholds.loc[state1][0])
    shapelet_ts2 = shapelet_representation(ts2,m_0=Thresholds.loc[state2][0])
    distance = cosine(shapelet_ts1,shapelet_ts2)
    return distance
