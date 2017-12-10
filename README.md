# Udacity_CarND_BehavioralCloning
Term1-Project3: Behavioral Cloning

## Project Goals

The primary goal of the project was to implement a CNN that can drive a
simulated car in a simulated environment by learning to predict the steering
angles.
Specifically, the project focued on the following tasks

* Using the simulator to collect data of good driving behavior
* build a CNN using Keras that predicts steering angles from images
* Train and validate the models from the created dataset
* Test the model by driving around the track in the simulator.

The project is termed successful if the car is able to drive around the track
one of the simulator without leaving the road.

## Contents

* Model.py - Contains the implementation of CNN and model generation code.
* model.h5 - The model generated by Model.py
* drive.py - Uses model.h5 to predict & feed steering angles to the simulator.
* video.mp4 - video recording of the final drive of the car around track one.

## Usage

### 1. Train and generate a model

```
python model.py
```
This generates __model.h5__

### 2. Feed steering angles to drive the car

First launch the simulator in autonomous mode and then start drive.py to inject
the predicted steering angles from model.h5 into the simulator

```
python drive.py model.h5
```

__NOTE:__ I had a powerful system with large memory resources to train the
model. In case you're seeing performance issues, you might want to modify the
model.py to use a generator to generated training data instead of storing it in
memory.

## Model & Training strategy

### Model used

The final implementation use the Nvidia CNN model to predict steering angles. It
consists of 5 convolution layers, 4 fully connected layers and kernels of sizes
5x5 & 3x3 with RELU as activation function, as shown below.

![nvidia cnn architecure]()

The Keras implementaion of the model accepts RGB input images (320x160) which is
then corpped (for region of interest i.e, the road) and normalized (divide by 
255 and subtract 0.5 per pixel) using __Cropping2D__ and __Lambda__ layers
provided by Keras.

LeNet model was used initially, and later replaced with this Nvidia model for
better results. More details in the model documentation section below.

### Overfitting reduction

The model was trained and validated with multiple different datasets to ensure
no over fitting. The model was tested by running it through the simulator in
different direction and scenerios ensuring that the vechicle always stayed on
the track. Addtionally, multiple experiments with dropout layers were done to
analyse and reduct overfitting.

The loss graphs in the section below shows the iterative improvements in the
model.

### Model parameter tuning

Since the __adam__ optimizer provided by Keras was used for training, no other
manual tuning was done.

### Training strategy

Training involved multiple iterations, with each iteration identifying the
failing conditions and adding more training data for the failed scenarios.

Before arriving at the final model & dataset, the training went through ~25-30
iterations. The multiple .h5 files corresponds to different iterations.

The final dataset constitues of 30000 captured images and these were augmented with
their flipped set to address driving in opposite direction. The dataset includes
the following scenarios

1. 4 laps of good driving in center lane at max speed
2. 4 laps of good driving in center lane at reduced speed (speed in autonomous mode)
3. multiple laps of recovering from road edges
4. multiple iterations of sections of curves with different markings
5. multiple iterations of sections with bright/sunny areas
6. 3 laps at reduced speed and lots of small steering corrections (more details
   in below sections)

## Model architecture and Training documentaitons

### Model design

The CNN models used for the project was not designed, rather, existing desings
were used & implemented. Initial model used the LeNet architecture with 2
convolution layers and 3 fully connected layers with 3x3 kernels, RELU
activation and adam optimizer. However, this model did not show good results
despite several kinds of training data although the train & validation loss
converged at low values. Below list shows the multiple iteration of this model with different
epochs & datasets and loss graphs of the last two models.

1. model-lenet.h5
2. model-lenet-augmented-data.h5
3. model-lenet-augmented-data-cropped.h5
4. model-lenet-augmented-data-cropped-multicam.h5
5. model-lenet-augmented-data-cropped-multicam-additional-data.h5
6. model-lenet-augmented-data-cropped-multicam-additional-data-2.h5
7. model-lenet-augmented-data-cropped-multicam-additional-data-3.h5
8. model-lenet-augmented-data-cropped-multicam-additional-data-4.h5

![leNet loss graph - 1]()
![leNet loss graph - 2]()

Visual analysis of this model while testing in the simulator showed that its able to predict 
steering angles well only in some cases like when in center of the road or without curves...etc, 
but the predictions weren't accurate in sections of curves, bright areas and road edge anomalies.
This made me feel the LeNet architecture with 2 levels of convolutions didn't
suffice to learn the features from the images to appropriately determine the
steering angles. Adding a few more layers of convolution seemed necessary.
Therefore instead of experimenting with my custom model, in the interest of
time, I decided to use Nvidia's CNN architecture to see if it yields better
results.

Nvidia's CNN model architecture, as described [here](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) 
consists of 10 hidden layers with 6 convolutions and 4 fully connected. 5x5
kernels are used between first 3 convlutions and 3x3 kernels between the last
two. RELU activation is used for all layers. The below picture represents the
CNN architecture.

![Nvidia cnn architecture]()

With the dataset already created for the previous model, this Nvidia's model
showed considerable improvement i.e, the car was able to drive much futher along
the track than with previous model with the same training data. However, this
also led the car outside of the road, specially at the curves. At this point the
dataset consisted of around 15000 images. The loss graphs depicted below also
showed signs of overfitting.

![Nvidia loss graph 1]()

Therefore, I collected more driving data for such scenarios. Further iterations
with additional data collected per iteration showed improvements, but weren't
completely successful. The loss graphs indicated the training required more
epochs. 

![Nvidia loss graph 2]()
![Nvidia loss graph 3]()
![Nvidia loss graph 4]()
![Nvidia loss graph 5]()
![Nvidia loss graph 6]()
![Nvidia loss graph 7]()

Finally, after 8 iterations of additional data with an accumulation of ~30000
images and 20 epochs, I arrived at a sufficiently good model that was able to
drive around the lap successfully without going off the road. The loss graph of
this model also showed stable convergence at low values for 20 epochs.

![Nvidia loss graph 8]()

I also additionally tested this model to drive around the track for multiple
laps in both directions and along with & without intermediate manual
interventions. The model was able to predict the steering angles to keep the car
always on the road and was also able to recover from edges/corners to get back
into cetner/safe driving section of the road. Since this sufficed the project
requirements, I concluded this as my final model.

#### Model Anomalies

Although I concluded the above as the final model, it didn't include any
droupout layers to reduce overfitting. So I tried two more iterations with the
finalized model with the same dataset, one with 3 20% dropout layers (one 
dropout layer after each convolution) and the second one with 2 20% dropout 
layers (one before the first convolution and the second after the last 
convolution). Both of these resulted in loss graphs as shown below with
validation error lower than the training error!

![Nvidia loss graph 9]()
![Nvidia loss graph 10]()

I wasn't able to interpret these results accurately. They indicated that model
was able to perform much better on validation set than on the training set. I
figured such results might happen in 2 scenarios

1. The split of validation set contained easier cases/images to predict than the
   training set.
2. The validation set different from the training set

The second case couldn't be possible because the validation & training set were
split as 20% & 80% accordingly from the same dataset. I couldn't confirm the
first case (I didn't have liberty of much time) by actually going over the
validation set images. Moreover, I hadn't updated the dataset since the last
finalized model.

Visual inspection of these models when run in the simulator showed that they
perfomred successfully for one lap, but went off the road during second or third
lap. Therefore, at this point I couldn't term these as good or bad models and
rather call them as unknown. Hence I still conclude my earlier finalized model
without any dropout layers for the submission.

I'll probably get back to analyse & learn about these anomalies when I get more
time.

### Dataset creation and training

The dataset is created by collecting images from the simulator which driving the
car in recording mode. The simulator provides facilities to dump the images. The
images captured correspond to front center, front left and front right cameras
as shown below. 

![center img 1]()
![center img 2]()
![center img 3]()

Each image is 320x160 in size.

The first set of data was created from good driving in center of the lane as
shown in the above images. This consisted of 4 laps of good driving data.

The next set of driving data included more curves and bright sections of the
lap.

![center curve 1]()
![center curve 2]()

In the further iterations of the model as described in the above sections, more
data were collected to represnt recovery from sides, driving parallel to the
lane marking, maintianing the center...etc scenarios.

![center recovery 1]()
![center reocvery 2]()
![left recovery 3]()

The dataset also included multiple laps & sections of driving at different
speeds (max & reduced). Since my driving inputs were provided from keyboard, a
noticible effect was the number of steering correction/keyboard inputs made at
higher & lower speeds. The initial data constituted of only high speed which
involved less keyboard input i.e, steering corrections and as result, the model
didn't provide more steering corrections required during the turns and this lead
the car off the road. After this observation, 4 laps driving data was collected
at reduced speeds (similar to the speed in autonomous mode) and also included
more keyboard inputs / steering corrections of smaller angles. This showed
showed a considerable improvement in the driving behaviour model.

Lastly, for every iterations of the model, more driving data was collected for
the sections in the lap for which the model failed in that iteration. The final
dataset consists of about 30000 images of these multiple scenarios.

#### Steering angle correction

The simulator dumps one steering angles for a set of center, left & right camera
image of an instance. This steering angle corresponds to the center images.
Therefore, in order to use the left & right images as well, the steering angles
were offset by a factore of +0.2 & -0.2 respectively and fed into the training
data.

#### Augmented data

Apart from the actual images captured as mentioned in the above section, these
images and its corresponding steering angels were flipped to account for driving
in opposite direction. Ideal scenario required to capture driving in the
opposite direction, however, for the simulator environment of track one, just
flipping the images sufficed.

## Results.

Here is the video recording of the final test drive in which the car
successfully drove around the track one for a little over 2 laps. (The below
video is fast forwarded 4x in the interest of time. video.mp4 contains the
original recording).


Here is the recording from some of the training iterations before arriving at
the final model.





