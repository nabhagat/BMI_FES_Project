# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 17:14:09 2016

@author: nikunj
"""
# Functions to be used with impulse recruitment curve.py

# <codecell>    
# Clear all variables - MATLAB equivalent 
def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]
#if __name__ == "__main__":
#    clear_all()
 
 

# <codecell>

def peak_baseline_normalize_force(force_response):
    # Low pass filter force response at 10Hz - Do not filter as it introduces distortion - Instead use SavitzkyGolay filter
    #b_lpf, a_lpf = sp.signal.butter(8, 5/float(Fs_force/2), 'low', analog=False)
    #force_response_filt = sp.signal.filtfilt(b_lpf,a_lpf,force_response, axis = 1)
    
    if len(force_response) == 0:
        peak_force_response = 0
        nan_removed_force_response_normalized = 0
        force_response_baseline_corrected = 0
    else:
        # Baseline correction and Normalize force_response to maximum for each trial 
        # Taking first sample as baseline voltage which is to be subtracted from entire signal
        force_response_baseline_corrected = np.zeros_like(force_response);
        force_response_normalized = np.zeros_like(force_response)
        moving_avg_filter_order = 3
        
        for trial_no,trial_force in enumerate(force_response):
            force_response_baseline_corrected[trial_no,:] = trial_force - trial_force[0]
            #force_response_baseline_corrected[trial_no,:] = savitzky_golay(force_response_baseline_corrected[trial_no,:], 9, 3)
            #force_response_baseline_corrected[trial_no,moving_avg_filter_order-1:] = moving_average(force_response_baseline_corrected[trial_no,:],moving_avg_filter_order)
            force_response_normalized[trial_no,:] = force_response_baseline_corrected[trial_no,:]/float(abs(force_response_baseline_corrected[trial_no,:]).max())
        
        nan_removed_force_response_normalized = force_response_normalized[~np.isnan(force_response_normalized).any(axis=1)]
        peak_force_response = force_response_baseline_corrected.max(axis=1)
    
    return peak_force_response, nan_removed_force_response_normalized, force_response_baseline_corrected

# <codecell>
#http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
def find_nearest(array,value_array):
    idx_array = []
    for u in value_array:
        idx_array.append((np.abs(array-u)).argmin())
    return idx_array

# <codecell>
def xcorr(x, y, k, normalize=True):

    n = x.shape[0]

    # initialize the output array
    out = np.empty((2 * k) + 1, dtype=np.double)
    lags = np.arange(-k, k + 1)

    # pre-compute E(x), E(y)
    mu_x = x.mean()
    mu_y = y.mean()

    # loop over lags
    for ii, lag in enumerate(lags):

        # use slice indexing to get 'shifted' views of the two input signals
        if lag < 0:
            xi = x[:lag]
            yi = y[-lag:]
        elif lag > 0:
            xi = x[:-lag]
            yi = y[lag:]
        else:
            xi = x
            yi = y

        # x - mu_x; y - mu_y
        xdiff = xi - mu_x
        ydiff = yi - mu_y

        # E[(x - mu_x) * (y - mu_y)]
        out[ii] = xdiff.dot(ydiff) / n

        # NB: xdiff.dot(ydiff) == (xdiff * ydiff).sum()

    if normalize:
        # E[(x - mu_x) * (y - mu_y)] / (sigma_x * sigma_y)
        out /=  np.std(x) * np.std(y)

    return lags, out

# <codecell> Filtering

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    #import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    
    return ret[n - 1:] / n
    
    
# <codecell>

# Model similar to tanh(x) - adopted from Freeman textbook

def f_IRC_2(u,c1,c2):
    return c1*abs((np.exp(c1*u)-1)/(np.exp(c2*u)+1))

# Take c3 = 1
def f_IRC_3(u,c1,c2,c3):
    return c1*abs((np.exp(c1*u)-1)/(np.exp(c2*u)+c3))

# Doesnot work
#def f_IRC1(u,c1):
#    return c1*abs((np.exp(c1*u)-1)/(np.exp(c1*u)+1))

def peak_muscle_gain(y_pred,Lambda):
    k=np.arange(1,y_pred.size+1,1)
    return y_pred*slope*(Lambda**k) + intercept
