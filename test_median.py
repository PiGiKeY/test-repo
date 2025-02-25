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