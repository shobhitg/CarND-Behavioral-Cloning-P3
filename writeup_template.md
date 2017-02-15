#**Traffic Sign Recognition**

**Behavrioal Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/nvidia-model.png "Model Visualization"
[image2]: ./images/sample-center-image.jpg "Grayscaling"
[image3]: ./images/recovery-1.jpg "Recovery Image"
[image4]: ./images/recovery-2.jpg "Recovery Image"
[image5]: ./images/recovery-3.jpg "Recovery Image"
[image6]: ./images/sample-frame.jpg "Normal Image"
[image7]: ./images/sample-frame-flipped.jpg "Flipped Image"

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model-udacity-track-1-complete.h5 containing a trained convolution neural network for Track 1
* model_worked_for_track2.h5 containing a trained convolution neural network for Track 2
* a writeup report sumamrizind the results.

#### 2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
For track 1
-----------
python drive.py model-udacity-track-1-complete.h5

For track 2
-----------
python drive.py model_worked_for_track2.h5
```

#### 3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model arcthiecture has been employed

I adopted the use of NVIDIA's neural network model as described in [https://arxiv.org/abs/1604.07316](https://arxiv.org/abs/1604.07316).

I had to improvise a little bit by adding some dropout layers to avoid overfitting.

The model includes RELU layers to introduce nonlinearity, and dropouts to help avoid overfitting. This can be seen in the function `create_model()`.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py `create_model()`). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code `split_validation()`). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py `train()`).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network that is known to produce good results for such usecases (NVIDIA as described above).

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

My objecttive was to qualize the loss for training and validation sets as much as possible.

To combat the overfitting, I modified the model to have more dropout layers that what NVIDIA mode prescribes.

Then I used the keras generators to add some augmentation for shearing and zooming.

The final step was to run the simulator to see how well the car was driving around track one.

There were a few spots where the vehicle fell off the track 1. To improve the driving behavior in these cases, I had to make use of left and right images with + and - 0.25 change in angles.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py `create_model()`) consisted of a convolution neural network with the following layers and layer sizes:

The summary below describes NVIDIA's model the best way:

```
|________________________________|_____________________|____________|_______________________________|
|Layer (type)                    | Output Shape        |  Param #   |  Connected to                 |
|================================|=====================|============|===============================|
|convolution2d_1 (Convolution2D) | (None, 31, 98, 24)  |  1824      |  convolution2d_input_1[0][0]  |
|________________________________|_____________________|____________|_______________________________|
|convolution2d_2 (Convolution2D) | (None, 14, 47, 36)  |  21636     |  convolution2d_1[0][0]        |
|________________________________|_____________________|____________|_______________________________|
|convolution2d_3 (Convolution2D) | (None, 5, 22, 48)   |  43248     |  convolution2d_2[0][0]        |
|________________________________|_____________________|____________|_______________________________|
|dropout_1 (Dropout)             | (None, 5, 22, 48)   |  0         |  convolution2d_3[0][0]        |
|________________________________|_____________________|____________|_______________________________|
|convolution2d_4 (Convolution2D) | (None, 3, 20, 64)   |  27712     |  dropout_1[0][0]              |
|________________________________|_____________________|____________|_______________________________|
|dropout_2 (Dropout)             | (None, 3, 20, 64)   |  0         |  convolution2d_4[0][0]        |
|________________________________|_____________________|____________|_______________________________|
|convolution2d_5 (Convolution2D) | (None, 1, 18, 64)   |  36928     |  dropout_2[0][0]              |
|________________________________|_____________________|____________|_______________________________|
|flatten_1 (Flatten)             | (None, 1152)        |  0         |  convolution2d_5[0][0]        |
|________________________________|_____________________|____________|_______________________________|
|dense_1 (Dense)                 | (None, 1164)        |  1342092   |  flatten_1[0][0]              |
|________________________________|_____________________|____________|_______________________________|
|dropout_3 (Dropout)             | (None, 1164)        |  0         |  dense_1[0][0]                |
|________________________________|_____________________|____________|_______________________________|
|dense_2 (Dense)                 | (None, 100)         |  116500    |  dropout_3[0][0]              |
|________________________________|_____________________|____________|_______________________________|
|dropout_4 (Dropout)             | (None, 100)         |  0         |  dense_2[0][0]                |
|________________________________|_____________________|____________|_______________________________|
|dense_3 (Dense)                 | (None, 50)          |  5050      |  dropout_4[0][0]              |
|________________________________|_____________________|____________|_______________________________|
|dense_4 (Dense)                 | (None, 10)          |  510       |  dense_3[0][0]                |
|________________________________|_____________________|____________|_______________________________|
|dense_5 (Dense)                 | (None, 1)           |  11        |  dense_4[0][0]                |
|================================|=====================|============|===============================|
Total params: 1,595,511
Trainable params: 1,595,511
Non-trainable params: 0
```


Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to make sharp steering angles when close to the curb.

These images show what a recovery looks like starting from near curb to middle of the road:

![alt text][image3]

![alt text][image4]

![alt text][image5]


To augment the data sat, I also flipped images and angles thinking that this would help neutralize the bias that one side would have over another. Not only that, I used flipped images as augmented data points.

For example, here is an image that has then been flipped:

![alt text][image6]

![alt text][image7]

Etc ....

After the collection process, I had 36k of data points. I then preprocessed this data by chopping off the top portion of each image by 40 pixels. And then resizing to 200 x 66 such that I could fit this into NVIDIA model.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was between 20 to 30 as evidenced by my `train()` function in model.py.

I used an adam optimizer so that manually training the learning rate wasn't necessary.


### Link to Youtube Videos (sped up versions so that they aren't boring)

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/LkG-V4fWfG0/0.jpg)](http://www.youtube.com/watch?v=LkG-V4fWfG0)

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/DvriT0sxlwg/0.jpg)](http://www.youtube.com/watch?v=DvriT0sxlwg)