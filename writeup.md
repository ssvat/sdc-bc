#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---
###Files Submitted 

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code 

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data from both udacity and simulator have been chosen to keep the vehicle driving on the road. 
For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to have the autonomous car finish the track 1.

My first step was to use a convolution neural network model similar to the NVIDIA architecture. 
I thought this model might be appropriate because it sucessfully deliver the solution.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 
In the beginning, the mean squared error of training and validation data is low but the car in autonomous mode did not drive well.
Also just in case of any overfitting, I have modified the model by adding a dropout of 25%.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network.
So there are 4 convolution layers and 3 fully connected layers.
I have normalization using a lambda layer and chose area of interest by cropping images.

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the udacity data and added a few laps on track one using center lane driving. 
Here is an example image of center lane driving:

![centre camera][image0.jpg]

I then recorded the vehicle from the left side, central and right sides of camera.  
This can compensate a recovery pass. The images from 3 cameras are shown below:

![centre camera][image0.jpg]
![right camera][image1.jpg]
![left camera][image2.jpg]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help generalize the model in right turns.
For example, here is an image that has then been flipped:

![image3][image3.jpg]
![flipped images][flip.png]

After the collection process, I had 6428 number of data points. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
The ideal number of epochs was 10 as evidenced by the video. I used an adam optimizer so that manually training the learning rate wasn't necessary.
I have had train several runs of 20,0000 number of data points for 2 weeks on the simulator. 
However, the data did not imporve the training so I removed the data and re-trained again with Udacity data after a few weeks. 
It may be because of a keyboard was used instead of a less sensitve mouse due to limited resources of personal computer system and accessories.   
A better mouse or a game controller can be used in the future to ensure the steering angles with smooth values.
Besides the numbers of epoch, size of batch, the camera correction factor has been experimented with different values, as well as different training network. 
I think the training data dominates the result. When I generate more data (conuterclockwise, clockwise, turns), 
the result did not improve and may become worse (loss >0.3), again due to the driving data, generated by the simulator under limited computer resources.
Therefore, the udacity data (loss <0.02) produce best result to finish the track 1 compared to local trainning data from me.   
Also, sometime I found even the loss is higher for a certain epoch but the car can drive longer and better. 
And sometimes even epoch is the same but the result is different from running on the same computer. 
The provided video is so far the best after comparing all of the stored results during each run.  
