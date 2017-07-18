# **Traffic Sign Recognition** 

## Writeup 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/class_count.png "Class Count Bar Chart"
[image2]: ./output_images/grayscale.png "Grayscaling"
[image3]: ./output_images/normalized.png "normalized"
[image4]: ./test_images/cleanstop7.jpg "Traffic Sign 1"
[image5]: ./test_images/curveRight4.png "Traffic Sign 2"
[image6]: ./test_images/no_entry.jpg "Traffic Sign 3"
[image7]: ./test_images/pedestrianCrossing101.png "Traffic Sign 4"
[image8]: ./test_images/stop1000.png "Traffic Sign 5"
[image9]: ./test_images/yield11.png "Traffic Sign 6"

### Writeup 

You're reading it! and here is a link to my [project code](https://github.com/yhung119/German_Traffic_Signs_Classification/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and pandas methods rather than hardcoding results manually.

I used pickle to load the data and get the basic summary of data using numpy.

* The size of training set is 34799
* The size of the validation set is 7842
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Below is the bar chart of number of images vs classes (0-42). 

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale to fit the LeNet model input. It is also easier for computer to intrepret since it has lower dimension.

Here is a list of example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data to rescale the images to range 0 to 1 instead of range from 0 to 255. 

Here is a list of examples after final preprocessing:

![alt text][image3]

The difference between the original data set and the augmented data set is that images is now within range of 0 to 1 and in grayscale.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GrayScale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x24 				|
| Dropout(keep_prob: 0.5) | |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x48  									|
| RELU     |
| Max pooling        | 2x2 stride, outputs 5x5x48   |
| Dropout(keep_prob: 0.5) | |
| Flatten            | outputs 1200  |
| Fully connected		| outputs 480 									|
| RELU | |
| Dropout(keep_prob: 0.5) | |
| Fully connected | outputs 84|
| RELU | |
| Dropout(keep_prob: 0.5) | |
| Softmax | outputs 43 |
 
 
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer with learning rate of 0.001, batch size of 128, and 100 epochs. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.00
* validation set accuracy of 0.997
* test set accuracy of 0.97

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I tried the original LeNet architecture, but the accuracy was only around 0.87 for training and even lower for testing/validation. I chose this architecture to do a POC that my data is fine.
* What were some problems with the initial architecture?
The accuracy was not good enough. Since it was used for digit recoginition, it probably need some improvement for a more complicated dataset such as the traffic signs. 
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
The validation accuracy was low but the training accuracy was high (over fitting) so I added dropout with keep_prob of 0.5 and cross_validation to train/validate the model.
* Which parameters were tuned? How were they adjusted and why?
I adjusted some hyperparamters on the architecture like stride size and depth of conv layer. I changed conv layer to 5x5 strides, and higher depth to learn more complicated/deeper representation. 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Conv layers works well because it is known for learning hierarchical representation and traffic signs are mostly composition of simple lines or circles. Dropout layer is useful when the model is overfitting.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]

The second and third images might be hard to regonize for the model because of the noisy background.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Dangerous curve to the right      		| Priority road   									| 
| Pedestrains     			| Roundabout mandatory 										|
| Stop					| Stop											|
| Stop	      		| Yield					 				|
| No passing			| No passing      							|
| Yield        | Yield                    |


The model was able to correctly guess 3 of the 6 traffic signs, which gives an accuracy of 50%. This compares favorably to the accuracy on the test set of 97%. 
The reason Prediction accuracy is low may be due to the background noise of images or lack of augmented data during training. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is really confident the sign is a Priority road sign, but the image is actually a Dangerous curve to the right sign. The probability is listed below.

| Prediction         	|     Probability       					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road | 0.999 |
| No vehicles | 0.00036723|
| Stop : 0.000243 |

For the second image, the model is almost certain about Roundabout mandatory but again, the sign was actually Pedestrians.
The probability is listed below.
| Prediction         	|     Probability       					| 
|:---------------------:|:---------------------------------------------:| 
|Roundabout mandatory | 0.8803 |
|Speed limit (30km/h) | 0.0321 |
|Priority road | 0.02117 |

For the third image, the model was almost certain about the Stop sign and it was a stop sign! The probability is listed below.

| Prediction         	|     Probability       					| 
|:---------------------:|:---------------------------------------------:| 
| Stop |  0.722 |
| Speed limit (30km/h) | 0.03529 |
| Yield | 0.0331737659872 |

For the fourth image, the model was about half certain it was a Yield sign but it was a stop sign. However, the second prediction was stop sign. 

| Prediction         	|     Probability       					| 
|:---------------------:|:---------------------------------------------:| 
| Yield |  0.611197054386 |
| Stop | 0.347433716059 |
| Priority road | 0.0279130265117 |

For the fifth image, the model was correct and almost certain that it was a No passing sign.

| Prediction         	|     Probability       					| 
|:---------------------:|:---------------------------------------------:| 
| No passing | 0.730805456638 |
| No passing for vehicles over 3.5 metric tons | 0.10007545352 |
| Yield | 0.0503109879792 | 

For the sixth image, the model was correct is really certain that it was a yield sign.

| Prediction         	|     Probability       					| 
|:---------------------:|:---------------------------------------------:| 
| Yield : 0.997839808464 |
| Speed limit (50km/h) | 0.00094722333597 |
| Speed limit (80km/h) | 0.000466124969535 |


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


