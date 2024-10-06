# Parameters

REGISTRATION_FOLDER = "registration"
MIN_CELL_SIZE = 40
INTERACTIVE = True
while True:
    response = input("Interactive? (y/n): ").lower()
    if response == 'y':
        INTERACTIVE = True
        break
    elif response == 'n':
        INTERACTIVE = False
        break
    else:
        print("Invalid input. Please enter 'y' or 'n'.")

# Imports

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import neurite as ne
import pickle
import cv2

# Detection

file_list = sorted(os.listdir(REGISTRATION_FOLDER))
all_cells = []
old_maps = {}
if INTERACTIVE:
    with open("cells.pickle", "rb") as f:
        old_cells = pickle.load(f)
        assert [f for f in file_list if f.endswith(".tif") and not f.endswith("_p.tif")] == [c['name'] for c in old_cells]
        for cells in old_cells:
            old_maps[cells['name']] = cells['cells']
else:
    if os.path.exists("cells.pickle"):
        with open("cells.pickle", "rb") as f:
            old_cells = pickle.load(f)
        for cells in old_cells:
            old_maps[cells['name']] = cells['cells']

Q = False
for file_name in file_list:
    if file_name.endswith(".tif") and not file_name.endswith("_p.tif"):
        cells = []
        file_path = os.path.join(REGISTRATION_FOLDER, file_name)
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)[:, :, 2]

        def detect_cells(image):
            _, image_thres = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((3, 3), np.uint8)
            image_morph = cv2.morphologyEx(image_thres, cv2.MORPH_OPEN, kernel, iterations=2)
            contours, hierarchy = cv2.findContours(image_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            contours = [c for c in contours if cv2.contourArea(c) > MIN_CELL_SIZE]
            cells = [(c.mean(axis=0)[0][1], c.mean(axis=0)[0][0]) for c in contours]
            return cells

        def show_cells(image, cells):
            image_copy = np.zeros((image.shape[0], image.shape[1], 3))
            image_copy[:, :, 0] = image / 255
            for cell in cells:
                cv2.circle(image_copy, (int(cell[1]), int(cell[0])), 10, (0, 0, 255), -1)
            plt.close()
            plt.figure(figsize=(14,7))
            plt.imshow(np.transpose(image_copy, (1, 0, 2)))
            plt.show(block=False)

        # read cells file and load existing points
        if file_name in old_maps:
            print("Load from old map for:", file_name)
            cells = old_maps[file_name]
        else:
            cells = detect_cells(image)
        if INTERACTIVE and not Q:
            show_cells(image, cells)

        # iterate and ask to add / remove
        if INTERACTIVE and not Q:
            while True:
                response = input("add cell (a x y), delete cell (d x y), new plot (p), next (n), or quit (q):\n").lower().strip()
                if response == "n":
                    break
                elif response == "q":
                    plt.close()
                    Q = True
                    break
                elif response == "p" or response == "":
                    show_cells(image, cells)
                else:
                    arguments = response.split()
                    if len(arguments) != 3:
                        print("Invalid input.")
                        continue
                    k, x, y = arguments
                    if not x.isdigit() or not y.isdigit():
                        print("Invalid input.")
                        continue
                    x, y = int(x), int(y)
                    if k == "a":
                        cells.append((x, y))
                        print("Added", x, y)
                    elif k == "d":
                        if len(cells) == 0:
                            print("No cell.")
                            continue
                        distance = lambda a, b: (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                        distance_array = [distance((x, y), (a, b)) for a, b in cells]                            
                        min_distance = min(distance_array)
                        if min_distance > 1800:
                            print("No near cell.")
                            continue
                        min_idx = distance_array.index(min_distance)
                        print("Removed", cells[min_idx][0], cells[min_idx][1])
                        cells.pop(min_idx)
                    else:
                        print("Invalid input.")
                        continue

        # add result
        detection_result = { "name": file_name, "shape": image.shape, "cells": cells }
        print(detection_result["name"], len(cells))
        all_cells.append(detection_result)

with open("cells.pickle", "wb") as f:
    pickle.dump(all_cells, f)

print("saved")