import pandas as pd
from scipy.ndimage import median_filter

# Sample DataFrame
data = {
    'A': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'B': [9, 8, 7, 6, 5, 4, 3, 2, 1]
}
df = pd.DataFrame(data)

# Apply median filter
df_filtered = df.apply(lambda x: median_filter(x, size=3))

print("Original DataFrame:")
print(df)
print("\nFiltered DataFrame:")
print(df_filtered)

import matplotlib.pyplot as plt
import numpy as np

# Przykładowe dane
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Tworzenie dwóch wykresów
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# Pierwszy wykres
ax1.plot(x, y1, label='sin(x)')
ax1.set_title('Wykres sin(x)')
ax1.legend()

# Drugi wykres
ax2.plot(x, y2, label='cos(x)', color='orange')
ax2.set_title('Wykres cos(x)')
ax2.legend()

# Wyświetlanie wykresów
plt.tight_layout()
plt.show()