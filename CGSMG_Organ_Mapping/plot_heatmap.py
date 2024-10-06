import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
import matplotlib.colors as mcolors
import numpy as np
import neurite as ne
import cv2

SIZE = (256, 128)
SIGMA = 8
WRITE = True
HEATMAP_ONLY = False

# Get Cells

# group_name -> cell_list
with open("transformed_cells_vxm.pickle", "rb") as f:
    transformed_cells = pickle.load(f)
grouped_cells = defaultdict(list)
for cells in transformed_cells:
    print(cells['name'], len(cells['transformed_cells']))
    grouped_cells[cells['name'].split("_")[0]] += [(x, y) for x, y in cells['transformed_cells'] if x and y]

colors = plt.cm.rainbow(np.linspace(0, 1, len(grouped_cells)))
styles = "+xXov^*s"

# Get Atlas

atlas_raw = np.load("atlas_vxm.npy")

atlas_raw = atlas_raw[0, :, :]
atlas_raw = cv2.resize(atlas_raw, SIZE[:: -1], interpolation=cv2.INTER_LINEAR)
atlas_raw = atlas_raw.T

atlas = (cv2.GaussianBlur(atlas_raw, (11, 11), 0) * 255).astype("uint8")
_, atlas = cv2.threshold(atlas, 60, 255, cv2.THRESH_BINARY)

# Plot

def get_heatmap(x, y, s):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=SIZE, range=[[0, SIZE[0]], [0, SIZE[1]]])
    heatmap = 100. * gaussian_filter(heatmap, sigma=s) / 8.
    return heatmap.T

def hex_to_rgb_normalized(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))

def generate_colormap(end_color, start_color="#ffffff"):
    colors = [hex_to_rgb_normalized(start_color), hex_to_rgb_normalized(end_color)]
    n_bins = [3]
    cmap_name = 'custom_div_cmap'
    cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    return cm

colormaps = ["#58595b", "#009bdf", "#703d00", "#b700d6", "#ff9933", "#008c40", "#55007a", "#d70010"]

for i, name in enumerate(grouped_cells):
    cells = grouped_cells[name]
    x, y = [c[0] for c in cells], [c[1] for c in cells]
    heatmap = get_heatmap(x, y, SIGMA)
    X, Y = np.meshgrid(np.linspace(0, 1, SIZE[0]), np.linspace(0, 1, SIZE[1]))
    heatmap_masked = np.ma.masked_where(atlas == 0, heatmap.copy())

    if not HEATMAP_ONLY:
        fig, axs = plt.subplots(2, 2, figsize=(16, 8))
        # scatter plot
        axs[0, 0].scatter(x, y, label=name, s=20, marker=styles[0], c="m")
        axs[0, 0].invert_yaxis()
        axs[0, 0].imshow(atlas_raw, cmap='gray')
        # heatmap plot
        axs[0, 1].imshow(heatmap, cmap=cm.jet)
        # atlas plot
        axs[1, 0].imshow(atlas, cmap='gray')
        # atlas bounded
        cmap = axs[1, 1].pcolormesh(X, Y, heatmap_masked, cmap=cm.jet)
        contour_lines = axs[1, 1].contour(X, Y, atlas, colors='k', linewidths=1)
        axs[1, 1].invert_yaxis()
        fig.colorbar(cmap)
    else:
        fig, ax = plt.subplots(figsize=(16, 8))
        cmap = ax.pcolormesh(X, Y, heatmap_masked, cmap=cm.jet)
        contour_lines = ax.contour(X, Y, atlas, colors='k', linewidths=1)
        ax.invert_yaxis()

    fig.suptitle(name)
    if not WRITE:
        plt.show()
    else:
        plt.savefig('./plot/' + name + '.eps', format='eps')
        plt.savefig('./plot/' + name + '.png')
    plt.close()
