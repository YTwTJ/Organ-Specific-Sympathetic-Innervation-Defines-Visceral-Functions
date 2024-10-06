# Parameters

ATLAS_NPY = "atlas_vxm.npy"
TRAINING_FOLDER = "registration"
SIZE = (256, 128)
NB_FEATURES = [
    [16, 32, 32, 32],             # encoder features
    [32, 32, 32, 32, 32, 16, 16]  # decoder features
]
LOSS_WEIGHTS = [1, 0.1]
BATCH_SIZE = 4
NB_EPOCHS = 25
STEPS_PER_EPOCH = 50
SAVED_MODEL = 'registration_vxm.h5'

while True:
    response = input("Train a new model? (y/n): ").lower()
    if response in ('y', 'n'):
        break
    else:
        print("Invalid input. Please enter 'y' or 'n'.")

if response == 'y':
    TRAIN_NEW_MODEL = True
else:
    TRAIN_NEW_MODEL = False

# Imports

import os, sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import voxelmorph as vxm
import neurite as ne
import pickle
import cv2
from tqdm import tqdm
import random

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Image Registration

file_list = sorted(os.listdir(TRAINING_FOLDER))
image_list = []

for file_name in file_list:
    if file_name.endswith("_p.tif"):
        file_path = os.path.join(TRAINING_FOLDER, file_name)
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image[:, :, 0], SIZE[:: -1], interpolation=cv2.INTER_LINEAR) # get the first tunnel
        image_list.append(image)

print(np.load(ATLAS_NPY).shape)
atlas = np.load(ATLAS_NPY)[2, :, :]
atlas = atlas / atlas.max() * 255
assert atlas.max() > 1
atlas = cv2.resize(atlas, SIZE[:: -1], interpolation=cv2.INTER_LINEAR)
image_list.append(atlas)

training = np.array(image_list).astype('float') / 255

# verify
print('training maximum value', training.max())

inshape = training.shape[1:]
vxm_model = vxm.networks.VxmDense(inshape, NB_FEATURES, int_steps=0)

print('input shape: ', ', '.join([str(t.shape) for t in vxm_model.inputs]))
print('output shape:', ', '.join([str(t.shape) for t in vxm_model.outputs]))

losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]

vxm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=losses, loss_weights=LOSS_WEIGHTS)

def vxm_data_generator(x_data, batch_size=BATCH_SIZE, exclude_atlas=False, val=None):
    vol_shape = x_data.shape[1:] # extract data shape
    ndims = len(vol_shape)
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    max_idx = x_data.shape[0]
    if exclude_atlas:
        max_idx -= 1

    while True:
        idx1 = np.random.randint(0, max_idx, size=batch_size)
        if val is not None:
            idx1 = [val]
        moving_images = x_data[idx1, ..., np.newaxis]
        idx2 = np.random.randint(0, max_idx, size=batch_size)
        if val is not None:
            idx2 = [-1]
        fixed_images = x_data[idx2, ..., np.newaxis]
        inputs = [moving_images, fixed_images]
        outputs = [fixed_images, zero_phi]
        
        yield (inputs, outputs)

# visualize
train_generator = vxm_data_generator(training)
in_sample, out_sample = next(train_generator)

images = [img[0, :, :, 0] for img in in_sample + out_sample] 
titles = ['moving', 'fixed', 'moved ground-truth (fixed)', 'zeros']

moving = training[0, ...][np.newaxis, ..., np.newaxis]
fixed = training[8, ...][np.newaxis, ..., np.newaxis]
val_input = [moving, fixed]

class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        val_pred = self.model.predict(val_input)
        images = [img[0, :, :, 0] for img in val_input + val_pred] 
        titles = ['moving', 'fixed', 'moved', 'flow']
        ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True, show=False)
        plt.savefig('./plot/registration_epoch_' + str(epoch) + '.png')
        plt.close()

if TRAIN_NEW_MODEL:
    hist = vxm_model.fit_generator(train_generator, epochs=NB_EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, verbose=2, callbacks=[MyCallback()])
    vxm_model.save_weights(SAVED_MODEL)
else:
    vxm_model.load_weights(SAVED_MODEL)

# val
val_pred = vxm_model.predict(val_input)

# Find Cells

with open("cells_p.pickle", "rb") as f:
    all_cells = pickle.load(f)

# make sure cell pkl is up-to-date
assert [f for f in file_list if f.endswith("_p.tif")] == [c['name'] for c in all_cells]

for cells in all_cells:
    new_cell_locations = []
    for cell in cells['cells']:
        x, y = cell[0] / cells['shape'][0] * SIZE[0], cell[1] / cells['shape'][1] * SIZE[1]
        if int(x) != 0 and int(y) != 0:
            new_cell_locations.append((int(x), int(y)))
    cells['new_cell_locations'] = new_cell_locations

# Transform

def generate_point_image(x, y):
    base = np.zeros(SIZE)
    base[x-4: x+4, y-4: y+4] = 1.
    return base

def get_transformed_point(x, y, i):
    val_generator = vxm_data_generator(training, batch_size = 1, val = i)
    val_input, _ = next(val_generator)
    val_pred = vxm_model.predict(val_input)
    point_wrapped = vxm.layers.SpatialTransformer()([generate_point_image(x, y)[np.newaxis, ..., np.newaxis], val_pred[1]])
    data = point_wrapped[0, :, :, 0]
    if np.max(data) <= 0:
        print(np.min(data), np.max(data))
        print(x, y)
        print(i)
        if x <= 5:
            return None
    assert np.max(data) > 0
    max_index = np.unravel_index(np.argmax(data), data.shape)
    return max_index

atlas_raw = np.load("atlas_vxm.npy")
atlas_raw = atlas_raw[0, :, :]
atlas_raw = cv2.resize(atlas_raw, SIZE[:: -1], interpolation=cv2.INTER_LINEAR)
atlas_raw = atlas_raw.T
atlas = (cv2.GaussianBlur(atlas_raw, (65, 65), 0) * 255).astype("uint8")
_, atlas = cv2.threshold(atlas, 40, 255, cv2.THRESH_BINARY)

def plot_transformed_images(i, base_cells, transformed_cells):
    val_generator = vxm_data_generator(training, batch_size = 1, val = i)
    val_input, _ = next(val_generator)
    val_pred = vxm_model.predict(val_input)
    val_wrapped = vxm.layers.SpatialTransformer()([val_input[0], val_pred[1]])

    base_image = val_input[0][0, :, :, 0].copy() * 255
    transformed_image = val_pred[0][0, :, :, 0].copy() * 255
    base_image_copy = np.zeros((base_image.shape[0], base_image.shape[1], 3))
    base_image_copy[:, :, 2] = base_image / 255
    base_image_copy[:, :, 1] = base_image / 255
    for cell in base_cells:
        cv2.circle(base_image_copy, (cell[1], cell[0]), 2, (255, 0, 255), -1)


    transformed_image_copy = np.zeros((transformed_image.shape[0], transformed_image.shape[1], 3))
    transformed_image_copy[:, :, 2] = atlas_raw.T
    transformed_image_copy[:, :, 1] = atlas_raw.T
    for cell in transformed_cells:
        cv2.circle(transformed_image_copy, (cell[1], cell[0]), 2, (255, 0, 255), -1)

    def save_image(image, name):
        plt.close()
        plt.imshow(image)
        plt.savefig('./plot/registration_' + name + '.png')

    save_image(base_image_copy, str(i) + "_moving")
    save_image(transformed_image_copy, str(i) + "_moved")

print("transforming...")
for i, cells in enumerate(tqdm(all_cells)):
    transformed_cells = []
    for cell in cells["new_cell_locations"]:
        p = get_transformed_point(cell[0], cell[1], i)
        if p:
            transformed_cells.append(p)
    cells['transformed_cells'] = transformed_cells

with open("transformed_cells_vxm.pickle", "wb") as f:
    pickle.dump(all_cells, f)

for i, cell in enumerate(all_cells):
    plot_transformed_images(i, cell["new_cell_locations"], cell["transformed_cells"])
