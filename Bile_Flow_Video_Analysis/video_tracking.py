# Bile flow video analysis - part 1 #

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import csv

start_time = 0
stop_time = 99999999999

row  = 10 # Enter total row number

file = 'xxx.mp4' # Enter video file name

cap = cv.VideoCapture(file)

backSub = cv.createBackgroundSubtractorKNN()
min_area_threshold = 15

# Calculate the desired starting frame
fps = cap.get(cv.CAP_PROP_FPS)
start_frame = int(start_time * fps)

# Set the video capture position to the desired starting frame
cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

refPt = []
cropping = False

def click_and_crop(event, x, y, flags, param):
    global refPt, cropping
    
    # If the left mouse button is clicked, record the starting (x, y) coordinates
    if event == cv.EVENT_LBUTTONDOWN:
        refPt = [[x, y]]
        cropping = True

    # If the left mouse button is released, record the ending (x, y) coordinates
    elif event == cv.EVENT_LBUTTONUP:
        refPt.append([x, y])
        cropping = False
        
        # Draw a rectangle around the region of interest (ROI)
        cv.rectangle(image, tuple(refPt[0]), tuple(refPt[1]), (0, 255, 0), 2)
        cv.imshow("image", image)

# Load the image
ret, image = cap.read()
clone = image.copy()

# Create a window and set the callback function for mouse events
cv.namedWindow("image")
cv.setMouseCallback("image", click_and_crop)

# Display the image
while True:
    cv.imshow("image", image)
    key = cv.waitKey(1) & 0xFF

    # Press 'r' to reset the cropping region
    if key == ord("r"):
        image = clone.copy()

    # Press 'c' to perform the crop
    elif key == ord("c"):
        cv.destroyAllWindows()
        break
    
# If two coordinates were recorded, perform the crop operation
if len(refPt) == 2:
    if refPt[0][1] > refPt[1][1]:
        temp = refPt[0][1]
        refPt[0][1] = refPt[1][1]
        refPt[1][1] = temp
    if refPt[0][0] > refPt[1][0]:
        temp = refPt[0][0]
        refPt[0][0] = refPt[1][0]
        refPt[1][0] = temp
    roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
frame_num = 0
cum_mask = np.zeros(1)
time = []
fill_area = []
timeout_count = 0
fluid_tip_cnt = None
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        timeout_count = 0
        frame_num +=1
        frame_crop = frame[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        
        fgMask = backSub.apply(frame_crop)
        num_labels, labels, stats, _ = cv.connectedComponentsWithStats(fgMask)
        
        fg_contours, _ = cv.findContours(fgMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        max_area = 0
        for contour in fg_contours:
            # Calculate the area of the contour
            area = cv.contourArea(contour)
            if area > max_area and area > min_area_threshold:
                max_area = area
                fluid_tip_cnt = contour
        
        mask = np.zeros_like(fgMask)
        for label in range(1, num_labels):
            # Check the area of the current component
            area = stats[label, cv.CC_STAT_AREA]
            # If the area is above the threshold, keep the component in the mask
            if area >= min_area_threshold:
                mask[labels == label] = True
        
        if cum_mask.any() == False:
            cum_mask = mask
        else:
            cum_mask = cum_mask + mask
            cum_mask[cum_mask>255] = 255
        
        _, bin_mask = cv.threshold(cum_mask, 50, 255, cv.THRESH_BINARY)
        num_labels_cum, labels_cum, stats_cum, _ = cv.connectedComponentsWithStats(bin_mask)
        total_white_pixels = 0
        for label in range(1, num_labels_cum):
            area_cum = stats_cum[label,cv.CC_STAT_AREA]
            total_white_pixels += area_cum
        time.append((frame_num-1)/fps)
        fill_area.append(total_white_pixels)
        if time[-1] + start_time >= stop_time:
            break
        
        if frame_num%100 == 0:
            # Display the resulting frame
            cv.imshow('Frame',frame_crop)
            cv.imshow('FG Mask', fgMask)
            cv.imshow('cumulative', bin_mask)
            
            # Press Q on keyboard to  exit
            if cv.waitKey(1) & 0xFF == ord('q'):  
                break
   # Break the loop
    else:
        timeout_count += 1
        if timeout_count > 5:
            break

# When everything done, release the video capture object
cap.release()

contours, _ = cv.findContours(bin_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
all_contour_points = []
for contour in contours:
    # Extend the list with the points of the contour
    all_contour_points.extend(contour)

# Convert the list to a NumPy array
all_contour_points = np.array(all_contour_points)

# Find the bounding rectangle that fits all contours
x_all, y_all, w_all, h_all = cv.boundingRect(all_contour_points)

# Draw the bounding rectangle on the image
cv.rectangle(frame_crop, (x_all, y_all), (x_all+w_all, y_all+h_all), (0, 255, 0), 2)

x,y,w,h = cv.boundingRect(fluid_tip_cnt)

cv.rectangle(frame_crop,(x,y),(x+w,y+h),(0,0,255),2)
cv.imshow('Frame', frame_crop)

curve_len = 0.70246483 # mm
straight_len = 18.5 # mm
len_per_row = 20 # mm
print("row", row)
total_fluid_length = len_per_row*(row-1)
print("total_fluid_length", total_fluid_length)
if row%2: # if row number is odd then fluid is moving in the right direction
    total_fluid_length += ((x + w - x_all)/w_all)*(2*curve_len + straight_len) - curve_len
else: # if row number is even then fluid is moving in the left direction
    total_fluid_length += ((x_all + w_all - x)/w_all)*(2*curve_len + straight_len) - curve_len

fluid_length = fill_area/max(fill_area) * total_fluid_length
plt.plot(time, 0.144 * fluid_length)
plt.xlabel('Time (s)')
plt.ylabel('Fluid Volume (uL)')
plt.show()

# Combine the lists into a 2D list
combined_list = list(zip(time, 0.144 * fluid_length))

# Save the combined list to a CSV file
with open(file+'.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(combined_list)
    