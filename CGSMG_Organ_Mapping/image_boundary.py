import pickle
import cv2
import neurite as ne
import numpy as np
import os

TOP_RATIO = 0.2
BOTTOM_RATIO = 0.2
RELATIVE = True

def process(file_path):
    raw_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)[:, :, 0]
    image = cv2.GaussianBlur(raw_image, (17, 17), 0).astype("uint8")
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    height, width = image.shape
    if RELATIVE:
        max_width = image.sum(axis=1).max() / 255
    else:
        max_width = width

    top_cur = 0
    while top_cur < height // 2:
        ratio = image[top_cur, :].sum() / 255 / max_width
        if ratio >= TOP_RATIO:
            break
        top_cur += 1
    down_cur = height - 1
    while down_cur > height // 2:
        ratio = image[down_cur, :].sum() / 255 / max_width
        if ratio >= BOTTOM_RATIO:
            break
        down_cur -= 1
    raw_image[: top_cur, :] = 0
    raw_image[down_cur: , :] = 0
    new_filepath = file_path + '_p.tif'
    new_tif = np.zeros((height, width, 3), dtype=np.uint8)
    new_tif[:, :, 0] = raw_image
    cv2.imwrite(new_filepath, new_tif)
    return top_cur, down_cur

with open("cells.pickle", "rb") as f:
    list_of_cell_data = pickle.load(f)

for cell_data in list_of_cell_data:
    y1, y2 = process('./registration/' + cell_data['name'])
    cell_data['name'] = cell_data['name'] + '_p.tif'
    cell_data['cells'] = [(y, x) for (y, x) in cell_data['cells'] if y >= y1 and y < y2]
    print(cell_data['name'], len(cell_data['cells']))

with open("cells_p.pickle", "wb") as f:
    pickle.dump(list_of_cell_data, f)

for file_name in os.listdir('training'):
    if file_name.endswith('.tif') and not file_name.endswith('_p.tif'):
        process('./training/' + file_name)
