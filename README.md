## Vehicle Detection

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


## Dataset

Here are links to the labeled data.
* [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip)
* [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)

These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the
[KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/),
and examples extracted from the project video itself.
Also,

* [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations)

could be used to augment the training data, but not used for this project yet.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example_good.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

[image_r1]: ./examples/result1.png
[image_r2]: ./examples/result2.png
[image_r3]: ./examples/result3.png
[image_r4]: ./examples/result4.png
[image_t1]: ./examples/target1.png
[image_t2]: ./examples/target2.png
[image_t3]: ./examples/target3.png
[image_t4]: ./examples/target4.png
[image_final]: ./examples/final.png

---

#### 1. Feature Extraction using HOG (Histogram of Oriented Gradients)

The code for this step is contained in `P5_training.py`, specifically
the details in `my_cv_tools.py` line 73.

First, the example of `vehicle` and `non-vehicle` images are shown
here.

![alt text][image1]

Second, I explored different color spaces and different
`skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and
`cells_per_block`).  I grabbed random images from each of the two
classes and displayed them to get a feel for what the `skimage.hog()`
output looks like.

Here is an example using the `HSV`, `LUV`, `HLS`, `YUV`, `YCrCb` color
space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)`
and `cells_per_block=(2, 2)`:

![alt text][image2]

In addition to HOG feature, I used
* resized feature : converting the chosen color space to 32x32x3 (3072)
* histgram : histgram of  the chosen color space 32x3 (96)
.

#### 2. HOG parameters choise

The code for this step is contained in `P5_training.py`, specifically
the details in `my_cv_tools.py` line 8. For `P5_prediction.py`, please
see line 152 where hog is computed only once and sub sampling technique
is used thereafter.

Although this visulization may help to understand the each feature
intuitively, what I understood so far is that the intuitively
interpretable feature does not always perform well.

After the trial and error by iterating all the color space, I picked
up 3 channels from 'YCrCb' color space. Also, [this
paper](https://pure.tue.nl/ws/files/3283178/Metis245392.pdf) shows the
YCrCb space works well when HOG feature is used.

The parameter for `orientations` is chosen as 9, because it can
capture unique directions including both horizontal and vertical
edges. I tied 12 though, they are symmetry and does not perform well
in terms of accuracy improvement. It actually merely increased feature
space.

The parameter for `pixels_per_cell` and `cells_per_block` is chonsen
to be 8 and 2, as it can reasonably capture the edge information after
trial and error.

#### 3. Training

The final feature is created by concatenating the resized feature,
histgram and HOG feature in 'YCrCb' color space. Then the feature is
normalized by standardization (average = 0, standard deviation = 1).

I trained a SVM with grid search (line 79). Kernel is chosen from
{linear, rbf}, and hyper paramter C and gamma are chosen. Grid search
was excuted to training data(80% of entire datset) in multi threads to
make it faster. Internally, grid search internally uses 3-fold cross
validation and choose the best performance parameter.

For the test performance (20% of dataset), I achieved 99.62% accuracy
to classify car and non-car.

### Sliding Window Search

#### 1. Parameter Tuning
The code is shown in `P5_prediction.py` line 291 and function of 'find-cars'.

I scaled the image so window search can be executed in different size.
The closer objects can be seen in larger pixels, while the farther
objects can be seen in smaller pixels. Thus, the window searches with
different scales deal with different area as shown below.

 Search Scales | 1.0 | 1.5 | 2.0 | 2.5 
:-----------:|:---:|:---:|:---:|:---:
 xstarts  | 600 | 600 | 600 | 600 
 xstops  | 1280 | 1280 | 1280 | 1280 
 ystarts  | 350 | 360 | 370 | 380 
 ystops  | 520 | 580 | 640 | 700

I added masking to x region too, as the algorithm sometimes capture
the cars in the opposite lane, which are distractions in highway
situation.

The target search areas are shown below.

scale 1.0              |    scale 1.5
:-------------------------:|:-------------------------:
![alt text][image_t1]      |      ![alt text][image_t2]

scale 2.0              |    scale 2.5
:-------------------------:|:-------------------------:
![alt text][image_t3]      |      ![alt text][image_t4]


#### 2. Result

Here is the result.

scale 1.0              |    scale 1.5
:-------------------------:|:-------------------------:
![alt text][image_r1]      |      ![alt text][image_r2]

scale 2.0              |    scale 2.5
:-------------------------:|:-------------------------:
![alt text][image_r3]      |      ![alt text][image_r4]

Here the resulting bounding boxes are drawn.
![alt text][image_final]

---



### Video Implementation

#### 1. Filter
Please see the code in `P5_prediction.py` line 319.

In video analytics, I applied filtering to avoid the false positive
data.  The past 6 frames are considered to compute filtering, meaning
24 results of window search is used as each fram has 4 different
version. The area is considered to be high-confidence detections where
multiple overlapping detections occur. Thus, if the area has more than
the certain threshold, in my case 10, the area is considered as
'blob'. Then, I constructed bounding rectangular that covers
'blob'. To realize this, I used `scipy.ndimage.measurements.label`.

#### 2. Result

Here's a [link to my video result](./project_video_out.mp4)

This video shows the car detection whil minimizing false positive.

Here's an example result showing the heatmap from a series of frames
of video, the result of `scipy.ndimage.measurements.label()` and the
bounding boxes then overlaid on the last frame of video:

---

### Discussion

#### 1. problems / issues of this method

* color space

It is very hard to find out the best color space based on intuitiion
or simple visualization shown above. It required me to iterate all the
color spaces. If a different situation (weather, night or day ...) is
given, I may need to tweak the color space again. Until I see the
result in the video stream, I can not guarantee the quality. There
should be evaluation metrics on the video analytics. In this way,
hyper parameters such as color space can be optimized.

* classification accuracy

When dealing with the binary classifcation of 64x64 images (car and
non-car), mostly I got 99.50 result by SVM. However, when it comes to
applying to video streming, the accuracy is not that good. I may
prepare for different training dataset which is close to the given
task. 

* masking

I used masking to ignore the left side, as they are distraction.  This
is based on the assumption that the current lane is measured in
different technique such as lane detection. If we are in the middle of
the lane, definitely we need to change the masking parameter.

* unknown objects

If the situation involves pedestrian or animals on the track, we need
to prepare for the different datsaet to detect them too to make the
model more advanced.

* prediction speed

In practical use, all the prediction needs to be done in real time.
Now, the computation takes way too long time. python is slow, but not
sure if it is going to be fast enough in C++. When it comes to deep
learning, we may need to think about how we can squeeze down the
model(squeezeNet) with keeping accuracy level.