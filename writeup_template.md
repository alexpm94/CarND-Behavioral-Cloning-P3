# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[image_3data]: ./examples/3_datasets.png
[image_merge]: ./examples/full_dataset.png
[image_right]: ./examples/right.jpg
[image_left]: ./examples/left.jpg
[image_center]: ./examples/center.jpg
[image_dave]: ./examples/dave2.png
[image_arch]: ./examples/architecture.png
[image_low]: ./examples/low.png
[image_theone]: ./examples/the_one.png
[image_final]: ./examples/final_arch.png



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 8 and 64 (model.py lines 108-122) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized and cropeed in the model using a Keras lambda layer (line 105-106). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 128 an 133). 
The model was trained and validated on different data sets to ensure that the model was not overfitting. (line 149)

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track, it was also tested counterwise, it performs well in both senses.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 147).

#### 4. Appropriate training data

I gathered training data using Udacity's simulator, I created three principal datasets.

1. Keeping the vehicle driving on the road. 
2. Driving counterwise.
3. Recovering from the left and right sides of the road.

From the three datasets listed above, I took the pictures of the left and right cameras as well, with a correction factor of -0.3 and 0.3 respectively for the steerin angles.

For details about how I created the training data, see the next section.

Here is the histogram of the 3 datasets. I decided to create 3 datasets in order to have almost the same quantity of images for steering to right and steering to left.

I also decided to flip the images from the center camera of the 1st and 3rd datasets. This is because when driving counterwise I just gathered data from one lap and I collected data from 2 laps when driving in the correct sense. 

![alt text][image_3data]

This is how it looks the 3 datasets merged.

![alt text][image_merge]

Here I show some examples from the recovering data.

![alt text][image_right]
Recovering right.

![alt text][image_left]
Recovering left.

![alt text][image_center]
After recovering from left.

I resized the images down to 50% size (from 160x320 to 80x160). It is important to mention that I firstly collected data driving at speed of 30 in the simulator, but it did not work well, so I gathered data driving between 15-20 of speed. 

I also normalized the data at the start of my neural network and cropped 25 pixels of the top and 10 of the bottom.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the NVIDIA DAVE2 teamd, I thought this model might be appropriate because it was actually implemented on a real self driving car.

![alt text][image_dave]

However it did not work as I hoped, so I modified. I deleted a convolution layer and resize to 80x160 instead of 66x200 as the DAVE2 architecture does. My final model looks as follows:

![alt text][image_arch]

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a very low mean squared error on the training and validation sets. I was impressed.

![alt text][image_low]

When I went in autonomous mode, I got an even bigger surprise, the model was very bad.

My initial model had some dropouts between the convolution layers and maxpooling too. I remove them and left only dropouts on the fully connected layers and maxpooling only at the end of the last convolution layer.

Then got the next result.

![alt text][image_theone]

The loss is 10 times bigger than the previous case, however, it did perform well on the track.
This means my previous model was overfitting. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, to improve this, I created the data set called "recovering", in which I saved images based on the curves.

#### 2. Final Model Architecture

[image_final]: ./examples/final_arch.png

After the collection process, I had 30332 number of data points.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 because it tended to vary the loss of validation, on 5 epchoch I got 0.0547 while at 10 epochs I got 0.0548. I used an adam optimizer so that manually training the learning rate wasn't necessary.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.
You can find 2 videos of the track.One keeping the road on the correct sense (correct_side.avi) and one counterwise (video_counterwise.avi).
