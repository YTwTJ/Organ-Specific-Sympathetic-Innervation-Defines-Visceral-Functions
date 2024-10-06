# Bile flow video analysis - part 2 #

import pandas as pd
import os

orig_filename = 'xxx.mp4.csv' # Enter analyzed csv file name
df = pd.read_csv(orig_filename, header=None).set_axis(['Time', 'Volume'], axis=1)

stim_start = 14
stim_stop = 30
highlight = 1
baseline = 5

# start and stop in secs
win_start = 60*(stim_start - baseline)
win_stop = 60*(stim_stop + baseline)
small_win_start = 60*stim_start
small_win_stop = 60*(stim_start+highlight)

step = 60
align = lambda t: df['Time'][df['Time'] > t].min()
time = [align(t) for t in range(win_start, win_stop + 1, step)]
volume = df.loc[df['Time'].isin(time), 'Volume'].tolist()
new_df = pd.DataFrame({'Time': time, 'Volume': volume})

# Plot cropped time range
import matplotlib.pyplot as plt
def plot(df):
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    plt.plot(x, y)

    # Then over-plot the desired segment in a different color (e.g., red)
    mask = (align(small_win_start) <= x) & (x <= align(small_win_stop))
    plt.plot(x[mask], y[mask], color='red')
    plt.axvline(x=small_win_start, color='red', linestyle='--')
    plt.axvline(x=small_win_stop, color='red', linestyle='--')

    plt.xlabel('Time (s)')
    plt.ylabel('Fluid Volume (uL)')
    plt.show()
    plt.clf()

# Normalize Volume by subtracting the first Volume value from all Volume values
new_df['Normalized_Volume'] = new_df['Volume'] - new_df['Volume'].iloc[0]

# Construct new file name
base_name, ext = os.path.splitext(orig_filename)
new_filename = 'nor_avg_' + base_name + ext

# Save the resampled and normalized data to a new CSV file
new_df.to_csv(new_filename, index=False)
plot(new_df[['Time', 'Normalized_Volume']])
