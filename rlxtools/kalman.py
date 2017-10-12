import numpy as np
from pykalman import KalmanFilter


def online_kf(x, cov, tm=1, om=1, burnout=40):
    """
    simple 1D online Kalman filter with one state

    :param x: signal
    :param cov: transition covariance
    :param tm: transition coef (the float of the 1x1 matrix)
    :param om: observation coef (the float of the 1x1 matrix)
    :param burnout: number of previous samples at each sample
    :return: the filtered signal
    """
    kf = KalmanFilter(transition_matrices=[[tm]], transition_covariance=[[cov]], observation_matrices=[[om]])
    f = []
    measurements = []
    for i,xi in enumerate(x):
        if i>burnout:
            m,c = kf.filter(measurements[-burnout:])
            f.append(m)
            m,c = kf.filter_update(m[-1], c[-1], xi)
        measurements.append(xi)
    return np.r_[f][:,-1,0]

def offline_kf(x, cov, tm=1, om=1):
    """
    simple 1D online Kalman filter with one state

    :param x: signal
    :param cov: transition covariance
    :param tm: transition coef (the float of the 1x1 matrix)
    :param om: observation coef (the float of the 1x1 matrix)
    :return: the filtered signal
    """
    kf = KalmanFilter(transition_matrices=[[tm]], transition_covariance=[[cov]], observation_matrices=[[om]])
    xk = kf.smooth(x)[0][:,0]
    return xk


def plot_kalman(x, s, cov, tm=1, om=1, kf_function=offline_kf):
    """
        x: measured signal
        s: base signal (without noise, the one that is to be recovered)
    """
    from statsmodels.tsa.stattools import acf
    from scipy import stats
    print cov
    xk = kf_function(x, cov, tm=tm, om=om)

    x = x[-len(xk):].copy()
    s = s[-len(xk):].copy()

    plt.figure(figsize=(20, 3))
    plt.subplot2grid((1, 5), (0, 0), colspan=4)
    plt.plot(xk, color="black", alpha=1, label="kalman")
    plt.plot(x, color="red", alpha=.5, label="signal")
    plt.plot(s, color="blue", alpha=.5, label="base signal")
    plt.legend()

    tit = "kalman covariance %.4f" % cov
    tit += ", err to noisy signal %.4f" % (np.mean(np.abs(x - xk)))
    tit += ", err to base signal %.4f" % (np.mean(np.abs(s - xk)))

    ar = acf(xk - x, 100)  # np.r_[[autocov(xk-x,i) for i in range(1,100)]]
    tit += ", autocorr residuals %.4f" % np.linalg.norm(ar)
    tit += ", ks-test to gaussian %.4s" % \
           stats.kstest(x - xk, stats.norm(loc=np.mean(x - xk), scale=np.std(x - xk)).cdf)[1]
    plt.title(tit)

    plt.subplot2grid((1, 5), (0, 4))
    plt.plot(ar[1:])
    plt.title("autocorrelation of residuals")