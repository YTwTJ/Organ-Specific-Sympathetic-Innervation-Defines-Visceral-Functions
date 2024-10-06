# Parameters

TRAINING_FOLDER = "training" # tif images for learning atlas
SIZE = (128, 64)
ENC_NF = [16, 32, 32, 32] # vxm encoder
DEC_NF = [32, 32, 32, 32, 32, 16, 16] # vxm decoder
LOSS_WEIGHTS = [0.5, 0.5, 1, 0.01] # vxm loss
EPOCHS = 25 # learning epoch
STEPS = 100 # learning steps each epoch
BATCH_SIZE = 1 # learning batch size
GAUSSIAN_BLUR = 9 # gaussian blur kernel
SAVE_ATLAS = "atlas_vxm.npy" # saved as 2d array, shape is SIZE, value is between 0 and 1
SAVED_MODEL = 'atlas_vxm.h5'

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

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import voxelmorph as vxm
import neurite as ne
import cv2
import random

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Learn

def load_training_data():
    file_list = sorted(os.listdir(TRAINING_FOLDER))
    image_list = []
    image_to_plot = []

    for file_name in file_list:
        if file_name.endswith("_p.tif"):
          file_path = os.path.join(TRAINING_FOLDER, file_name)
          image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
          image = cv2.resize(image[:, :, 0], SIZE[:: -1], interpolation=cv2.INTER_LINEAR) # get the blue tunnel
          image_to_plot.append(image)
          image_list.append(image)

    print("load number of images:", len(image_list))
    print("display training data example...")
    return np.array(image_list).astype('float')/255

training = load_training_data()

tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)

vol_shape = training.shape[1: ]
x_train = training[..., np.newaxis]

def template_gen(x, batch_size):
    vol_shape = list(x.shape[1:-1])
    zero = np.zeros([batch_size] + vol_shape + [2])

    while True:
        idx = np.random.randint(0, x.shape[0], batch_size)
        img = x[idx, ...]
        inputs = [img]
        outputs = [img, zero, zero, zero]
        yield inputs, outputs

model = vxm.networks.TemplateCreation(vol_shape, nb_unet_features=[ENC_NF, DEC_NF])

image_loss_func = vxm.losses.MSE().loss
neg_loss_func = lambda _, y_pred: image_loss_func(model.references.atlas_tensor, y_pred)
losses = [image_loss_func, neg_loss_func, vxm.losses.MSE().loss, vxm.losses.Grad('l2', loss_mult=2).loss]

model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=losses, loss_weights=LOSS_WEIGHTS)

def save_image(image, name):
    plt.close()
    plt.imshow(image, cmap='gray')
    plt.savefig('./plot/atlas_' + name + '.png')

def save_images(images, tag):
    for i, image in enumerate(images):
        save_image(image[:, :, 0], str(i) + '_' + tag)

if TRAIN_NEW_MODEL:
    model.set_atlas(x_train.mean(axis=0)[np.newaxis, ...])
    # train model
    gen = template_gen(x_train, batch_size=BATCH_SIZE)
    hist = model.fit(gen, epochs=EPOCHS, steps_per_epoch=STEPS, verbose=2)

    model.save_weights(SAVED_MODEL)
else:
    model.load_weights(SAVED_MODEL)

# val
val_pred = model.predict([x_train])
save_images(x_train, "train")
save_images(val_pred[0], "pred")

# visualize learned atlas
atlas = model.references.atlas_layer.get_weights()[0][np.newaxis, ...][0, :, :, 0]
atlas = np.clip(atlas, 0, 255)
atlas_256 = (cv2.GaussianBlur(atlas, (GAUSSIAN_BLUR, GAUSSIAN_BLUR), 0) * 255).astype("uint8")
_, atlas_thres_b = cv2.threshold(atlas_256, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, atlas_thres_z = cv2.threshold(atlas_256, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
atlas_array = [atlas, atlas_256, atlas_thres_z, atlas_thres_b]

save_image(atlas, 'result')

np.save("atlas_vxm.npy", np.array(atlas_array))

print("save atlas...")
ne.plot.slices(atlas_array)
