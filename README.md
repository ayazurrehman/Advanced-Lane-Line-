##Advanced Lane Finding

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg "Raw"
[image1a]: ./undistorted/calibration1.jpg "Undistorted"
[image2]: ./examples/straight_line1.jpg "Road Transformed"
[image3]: ./examples/binary_threshold.jpg "Binary Example"
[image4]: ./examples/transformed.jpg "Transformed"
[image4a]: ./examples/test3.jpg "Raw Image"
[image6]: ./examples/lane_identified.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

You're reading it!

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in "./CarND-Adv-Lane-Lines.ipynb" and it is part of the function `Cal_undist()`

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]
![alt text][image1a]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps notebook located in "./CarND-Adv-Lane-Lines.ipynb" and it is part of the function `binary_thresholding()`).  Here's an example of my output for this step.

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in notebook located in "./CarND-Adv-Lane-Lines.ipynb" under the section `Perspective Transform`.  The `warp()` function takes as inputs an image (`img`).  I chose to hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[214,717],
    [1096,717],
    [689,451],
    [596,451]])
    
dst = np.float32(
    [[310,717],
    [880,717],
    [880,10],
    [310,10]])

```

I verified that my perspective transform was working and an example can be seen below.

![alt text][image4]
![alt text][image4a]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

When I was able to create a thresholded binary image, I was able to identify the white pixels relevant to the lane lines using the histogram and sliding window approach as explained in the comments of the funtion `sliding_window`.
Using numpy, I was able to find out the coefficients for a second order polynomial that would fit the white pixels relevant to the lane lines.

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the section `Radius of Curvature` and the `Position of Vehicle` section of the python notebook and the comments explain the implementation.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this in the section `Finding lane lines using sliding window approach` in the function `sliding_window`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My approach handles every frame of the video as a single image and doesn't utilize the sequence of images and results of one frame in the following frame. This could cause problems if in a particular frame there is too much light and it is difficult to identify the lane lines. This can be resolved by using the values of other frames and averaging it to see if there is a drastic change in the predictions of a frame so it could be ignored. 

