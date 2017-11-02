**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/test1_heatmap_labels.png
[image4]: ./output_images/test2_heatmap_labels.png
[image5]: ./output_images/test3_heatmap_labels.png
[image6]: ./output_images/test4_heatmap_labels.png
[image7]: ./output_images/test5_heatmap_labels.png
[image8]: ./output_images/test6_heatmap_labels.png
[video1]: ./output_images/project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points 

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

## Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

![alt text][image2]

Here is an example using the `YCrCb` color space HOG parameters such as orient for the HOG features , and parameters used for spatial features and color histogram features.  Listed below were the training parameters used.  I got a good accuracy of 0.9942 on an SVM classifier.
{'X_scaler': StandardScaler(copy=True, with_mean=True, with_std=True),
 'cell_per_block': 2,
 'color_space': 'YCrCb',
 'hist_bins': 16,
 'hist_feat': True,
 'hog_channel': 'ALL',
 'hog_feat': True,
 'orient': 12,
 'pix_per_cell': 8,
 'spatial_feat': True,
 'spatial_size': (16, 16),
 'svc': LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
      intercept_scaling=1, loss='squared_hinge', max_iter=1000,
      multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
      verbose=0),
 'validation_accuracy': 0.99429999999999996,
 'xy_overlap': (0.8, 0.8),
 'xy_window': (64, 64),
 'y_start_stop': [400, 640]}


#### 2. Explain how you settled on your final choice of HOG parameters.

I experimented a lot on the HOG parameters as this was a feature I used for the SVM classifier.  The orient parameter was particularly important as I would find that if I increased the orientations from 8 to 12 then I could end up improving the predictions which ultimately resulted in more bounding boxes around the vehicles.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using a combination of the HOG features, spatial bin features, and color histogram features.  All three features were raveled to 1-D vectors and concatenated into a single feature vector.  I made sure to normalize this vector since the three individual features contained different magnitudes of values which can bias the results.  I got a good validation prediction with the parameters stated earlier.  With a prediction over 0.92 or so would yield some decent predictions but below that there would be some false positive detections.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search using a sliding window with an overlap of 80% of the window in both x and y directions.  I chose this higher overlap due to a desire to gather more overlapping bounding boxes to help strengthen the heatmap and labeling process.
In addition, I employed the use of the following scales:  scales = (0.85, 1.0, 1.2, 1.5, 1.7, 2.0)
Initially I tried scaling way below 0.95 in the range of 0.3, 0.5, etc., but I found these introduced a lot of detection of false positives and so I ended up staying at ranges above 0.85.  I had also experimented with scale values greater than 2.3 but I found these did not make much difference in the results and so I left them out.  I increased the orient to 12 in order to obtain more features for training from the HOG features.  The pixels_per_cell was left at 8 and I did not need to increase that.  For the spatial bin I did experiment with (32, 32) but in the end I used (16, 16).  

During the search for cars the I restricted the search area in the y-range between 400 and 640 pixels.  In particular the 0-400px range was excluded because we do not expect cars in that range.  This helped to reduce the search time which was also important since I used an xy_overlap window of (0.8, 0.8) and so there were many overlapping windows which would increase the time to search.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched using all three channels of the YCrCb.  I utilized HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  For optimization I ran it using a LinearSVC classifier which was fit against the car and notcar datasets provided.  Note that I augmented through duplication the set of car data from existing car data such the car set equaled the number of notcar set.  I chose YCrCb as it seemed to perform well as mentioned earlier.  This 

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap.  I maintained the set of heatmaps for each of the last 8 frames to aid in the a smoothing operation which took the set of bounding box points and computed the means.  I then thresholded that mean map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

There were some challenges in selecting the right colorspace which took some a good deal of experimentation.  I tried everything from 3D plots of the colorspaces and tried HSV, YUV, LUV, and YCrCb colorspaces.  I found I could get a reasonable set of detections using the H-channel of HSV but it had some difficulty detecting the white car at times and of course there were false positives both on the road to the side of the road.
I found another reasonable set of detections with YCrCb and I started with an 8 orient setting for HOG detection with pix_percell = 16.  These were not producing a sufficient number of detection boxes for some of the vehicles so I increased orient to 12 and this helped to aid in detection because it captured more features that a car would typically contain vs noncar patches.  I had also used a spatial_size of (32, 32) at first but found I got even better results at (16, 16) and so that's what I stayed with.
For overlapping I had started at (0.5, 0.5) but eventually moved upwards to (0.8, 0.8).  I even played with the non-square regions hoping to get more overlap in the vertical direction vs just the horizontal.  It worked somewhat but in the end it didn't make any difference.  A (0.8, 0.8) overlap worked quite well though I was always concerned of it resulting in more false positive detections which can happen and is something to keep in mind.  I felt the added detections outweighed the false positives and that I could address false positives more in the thresholding of the heatmaps.

With respect to heatmaps I chose to implement a FIFO queue to act as a buffer of the most recent bounding boxes which I could then average over to try and get a smoother bounding.  I still thresholded the result and at first tried to stay with a lower threshold of 2 but I found it necessary to increase this to 4. Any higher and I ran the risk the misidentification of the number of vehicles.
