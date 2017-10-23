#**Behavioral Cloning** 

##Submission By Narayanan Raghuvaran


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

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 A video recording of the car navigating track1 using drive.py
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

I started with the nVidia Architecture, I then read their paper.

Training strategy consisted of increasing the sample size from what udacity provided me, which was around 6k samples
I almost tripled it without additional driving by using the images from the left and right car cameras, this
proved invaluable as the center camera mostly provided 0 steering angle whereas the left and right cameras
came with correction steering angles which was non zero

I also did image pre processing as part of my training strategy

I added drop outs , used standard Lambda normalizations etc 


####1. An appropriate model architecture has been employed

My model consists of (from nVidia Paper)
1. Image normalization (model.py line 131)
2. Followed by 3 convolution layers with 3 X 3 filter model.py lines 132, 134, 136. All of them uses
ELU activation function which performs better
3. This was followed by 3 convolution layers with 5 X 5 filet model.py lines 138, 140. Both of them use ELU activation
function
4,. After flattening the data , it is fed to
5. 3 fully connected layers using ELU activation function, model.py lines 143 to 147

This multi layer network has sufficient dropouts and L2 regulariations to prevent over fitting
a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes ELU layers to introduce nonlinearity (code line 132,134,136,138), and the data is normalized in the model using a Keras lambda layer (code line 131). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 133,135,137,139,142). 
The model has L2 regularizations to reduce overfitting lines 

 The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 165).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Recovering from the left and right sides of the road was achieved by
adding the left and right camera images to the sample data

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the one proposed by nVidia in their research 
paper

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I ran training on data from all three camera angles and also flipped the image using a random probability between 0 and 1 (o.5 boundary) to add images fro driving in the other direction

To combat the overfitting, I modified the model with Dropouts and L2 regularizations

 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell went into the water or could not negtiate a sharp turn, so I had train the model with
more epochs so that mse < 0.3

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 128-150) consisted of a convolution neural network with the following layers and layer sizes
1. Lamda data normalization with 0 mean and range -0.5 to + 0.5
2. Conv2D 5 x 5 filter , ELU activation, L2 regularizer
3. Dropout
4. Conv2D 5 x 5 filter , ELU activation, L2 regularizer
5. Dropout
6. Conv2D 5 x 5 filter , ELU activation, L2 regularizer
7. Dropout
8. Conv2D 3 X 3 filter , ELU activation, L2 regularizer
9. Dropout
10.Conv2D 3 X 3 filter , ELU activation, L2 regularizer
11.Dropout, Flatten
12.Fully connected , ELU activation, L2 regularizer
13. Fully connected , ELU activation, L2 regularizer
14. Fully connected , ELU activation, L2 regularizer
15 Fully connected linear activation
Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)



####3. Creation of the Training Set & Training Process

I used the data set provided by Udacity.


I started with the basic nVidia architecture and used Udacity's supplied training data and ran the training simulation
This ended horribly as the car would wander off the road into the woods and ditch.

Something was wrong, I then remembered Udacity's image preprocessing notes in the videos. So I added image preprocessing
by cropping the image, and randomly adjusting brighness

I then ran the training again and ran the autonomous mode, this time the car could not 
navigate the sharp turns , the car could not navigate the first turn,
hmm the car prefers to drive straight.  The car has not learnt to turn

I looked at the driving_logs csv file and predominantly the center camera images had 0 steering angle
I also remembered the udacity video about left and right camera images that can be included 
with a correction factor
I augmented the data by including the images from the left and right cameras and adjusting the steering angle for each with
a correction factor

Also based on nVidia paper I converted the image to YUV format. Repeated the train/autonomous mode and much betterSimilarly cv2 output BGR format, converted to YUV, before sending to the model

Then I trained the model again , but still the car would wander off into the woods

So I reflected on what was going, the model architecture is pretty simple, so something is wrong elsewhere. For the first time I looked at drive.py, it then struck me that I have to do the same image pre processing in drive.py, before we 
predict the steering angle

So I cropped the image, resized it to the model's preferred shape i.e. 200, 66 and converted the color schema to YUV
Now I think I had a pretty good design. I trained the model using udacity's dataset. The validation error started at 0.52 and kept dropping , at about epoch 15,it was 0.30.I felt much better

I then ran drive.py with the new model. And the car could complete the laps on track 1 effortlessly. 

I am now working on track2. Here I have adjusted drive.py throttle to be dependent on steering angle and speed
(Steep turns and hilly track)

throttle = 0.65 - (steering_angle)**2/ 2   -  (current_speed/25)**2

Also steering_angle being fed to the controller is 1.2 * steering_angle to accentuate sharper turns.
This has taken me to the half way point for track2

The model works on track1 

Before feeding the data to the model, I had shuffled the training set before each epoch

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 15 as after that the validation error drop per epoch was very
negligible. I used an adam optimizer so that manually training the learning rate wasn't necessary.
