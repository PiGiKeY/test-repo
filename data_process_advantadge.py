import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, decimate
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

# Projektowanie filtra Butterwortha dolnoprzepustowego
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Parametry filtra
cutoff = 5  # Pożądana częstotliwość odcięcia filtra, Hz
fs=1000
order = 6     # Rząd filtra


df = pd.read_csv("4140_sily_short.csv", sep=';')

# Wyświetlenie zawartości DataFrame
#df=df.drop(index=0)
#times=df["t"]

times=df["Time (s)"].to_numpy()
forceX=df["Force-X (N)"]

# Decimate the signal
decimation_factor = 10  # Example decimation factor
dtimes=decimate(times, decimation_factor)
dfX = decimate(forceX, decimation_factor)

dforce=pd.DataFrame(dfX, columns=['Force-X (N)'])
# Obliczanie różnicy procentowej
dforce['Pct_Change'] = dforce["Force-X (N)"].pct_change(fill_method='pad') * 100 

# Definiowanie warunku procentowego (np. zmiana większa niż 10%)
condition = dforce['Pct_Change'].abs() > 0

# Apply the condition and compute the moving average
def conditional_moving_average(values, condition, window):
    filtered_values = values.where(condition, np.nan)
    return filtered_values.rolling(window=window, min_periods=1).mean()

# Compute the moving average with a window of 3
dforce["Fx avg (N)"] = conditional_moving_average(dforce['Force-X (N)'], condition, window=101)

# Compute the median filter
dforce["Fx med (N)"] = median_filter(dforce['Force-X (N)'], size=101)

# Apply a percentile filter
def percentile_filter(data, window_size, percentile):
    return data.rolling(window=window_size, center=True).apply(lambda x: np.percentile(x, percentile), raw=True)

# Compute the 90th percentile filter with a window of 801
dforce["Fx perc 90 (N)"] = percentile_filter(dforce['Force-X (N)'], window_size=101, percentile=50)

# Apply Savitzky-Golay filter
window_size = 101  # Window size (must be odd)
poly_order = 2    # Polynomial order
dforce["Fx SG (N)"] = savgol_filter(dforce['Force-X (N)'], window_size, poly_order)


SGforce=savgol_filter(dforce['Force-X (N)'],window_size,poly_order)
# Zastosowanie filtra dolnoprzepustowego
dforce["Filtered_forceX (N)"] = lowpass_filter(dforce["Force-X (N)"], cutoff, fs, order)

fig, axs =plt.subplots(2,2, figsize=(10,8))
axs[0,0].plot(times, forceX, label='Raw Data')
axs[0,0].plot(dtimes, dforce["Fx avg (N)"], label='Moving Average')
axs[0,0].set_xlabel('time (s)')
axs[0,0].set_ylabel('Force (N)')
axs[0,0].legend()

axs[0,1].plot(times, forceX, label='Raw Data')
axs[0,1].plot(dtimes, dforce["Fx med (N)"], label='Median')
axs[0,1].set_xlabel('time (s)')
axs[0,1].set_ylabel('Force (N)')
axs[0,1].legend()

axs[1,0].plot(times, forceX, label='Raw Data')
axs[1,0].plot(dtimes, dforce["Fx SG (N)"], label='Savitzky-Golay')
axs[1,0].set_xlabel('time (s)')
axs[1,0].set_ylabel('Force (N)')
axs[1,0].legend()

axs[1,1].plot(times, forceX, label='Raw Data')
axs[1,1].plot(dtimes, dforce["Filtered_forceX (N)"], label='Butterworth Lowpass')
axs[1,1].set_xlabel('time (s)')
axs[1,1].set_ylabel('Force (N)')
axs[1,1].legend()




#plt.plot(times, forceX["Fx perc 90 (N)"], label='90th Percentile')
#plt.plot(dtimes, SGforce, label='Savitzky-Golay')
#plt.plot(dtimes, dforce, label='Decimated')

plt.tight_layout()
plt.show()