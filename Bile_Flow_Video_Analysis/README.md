# Bile Flow Video Analysis

## Environment Setup
```
# set up python dependencies
conda env create -f environment-video.yml

# make sure to activate conda environment before running any python script
conda activate video
```

## Run video tracking: generate a time frame-fluid volume csv file
Check the code input: total row number
Run `video_tracking.py`
* draw ROI for video analysis; enter 'c' for proceeding, 'r' for re-draw

## Normalize and plot the first frame data every minute
Run `python plot.py`
