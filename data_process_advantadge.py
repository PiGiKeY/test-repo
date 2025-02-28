import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

# Projektowanie filtra Butterwortha dolnoprzepustowego
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Parametry filtra
cutoff = 5  # Pożądana częstotliwość odcięcia filtra, Hz
fs=9000
order = 6     # Rząd filtra


df = pd.read_csv("4140_sily_short.csv", sep=';')

# Wyświetlenie zawartości DataFrame
#df=df.drop(index=0)
#times=df["t"]

times=df["Time (s)"].to_numpy()
forceX=df["Force-X (N)"]
#forceX=df["Force-Y (N)"].to_numpy()
forceX=forceX.to_frame()

# Obliczanie różnicy procentowej
forceX['Pct_Change'] = forceX['Force-X (N)'].pct_change(fill_method='pad') * 100 

# Definiowanie warunku procentowego (np. zmiana większa niż 10%)
condition = forceX['Pct_Change'].abs() > 0

# Apply the condition and compute the moving average
def conditional_moving_average(values, condition, window):
    filtered_values = values.where(condition, np.nan)
    return filtered_values.rolling(window=window, min_periods=1).mean()

# Compute the moving average with a window of 3
forceX["Fx avg (N)"] = conditional_moving_average(forceX['Force-X (N)'], condition, window=801)

# Compute the median filter
forceX["Fx med (N)"] = median_filter(forceX['Force-X (N)'], size=801)

# Apply a percentile filter
def percentile_filter(data, window_size, percentile):
    return data.rolling(window=window_size, center=True).apply(lambda x: np.percentile(x, percentile), raw=True)

# Compute the 90th percentile filter with a window of 801
forceX["Fx perc 90 (N)"] = percentile_filter(forceX['Force-X (N)'], window_size=801, percentile=50)

# Apply Savitzky-Golay filter
window_size = 11  # Window size (must be odd)
poly_order = 3    # Polynomial order
y_smooth = savgol_filter(y, window_size, poly_order)

# Zastosowanie filtra dolnoprzepustowego
#forceX["Filtered_forceX (N)"] = lowpass_filter(forceX["Force-X (N)"], cutoff, fs, order)

print(forceX)
plt.plot(times, forceX["Force-X (N)"], label='Raw Data')
plt.plot(times, forceX["Fx avg (N)"], label='Moving Average')
plt.plot(times, forceX["Fx med (N)"], label='Median')
plt.plot(times, forceX["Fx perc 90 (N)"], label='90th Percentile')
plt.xlabel('time (s)')
plt.ylabel('Force (N)')
plt.legend()
plt.show()