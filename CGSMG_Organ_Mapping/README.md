# CGSMG - Organ mapping

## Environment Setup
```
# set up python dependencies
conda env create -f environment.yml

# make sure to activate conda environment before running any python script
conda activate tf_gpu
```

## Cell Detection
```
# interactive cell detection
python detect_cells.py
```

## Learn Atlas
```
# create a directory for plots
mkdir plot

# apply boundaries to data
python image_boundary.py

# save learned atlas in atlas_vxm.npy
python generate_atlas.py
```

## Image Registration
```
# registration
python image_registration.py

# plot heatmap
python plot_heatmap.py
```
