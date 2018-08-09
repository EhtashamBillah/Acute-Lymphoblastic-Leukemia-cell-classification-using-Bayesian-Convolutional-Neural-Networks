# Acute Lymphoblastic Leukemia cell classification from microscopic images using Bayesian Convolutional Neural Networks
In this project, we deploy the Bayesian Convolution Neural Networks (BCNN), proposed
by Gal and Ghahramani [2015] to classify microscopic images of blood samples (lymphocyte
cells). The data contains 260 microscopic images of cancerous and non-cancerous lymphocyte
cells. We experiment with different network structures to obtain the model that return
lowest error rate in classifying the images. We estimate the uncertainty for the predictions
made by the models which in turn can assist a doctor in better decision making. The
Stochastic Regularization Technique (SRT), popularly known as Dropout is utilized in the
BCNN structure to obtain the Bayesian interpretation.

---

## 1. Project Description:

We used the R programming language to implement Bayesian Convolutional Neural Networks in our image data. The Integrated
Development Environment (IDE) we utilized for programming was RStudio. The Deep Learning framework was TensorFlow and the Application Programming Interface
(API) on top of Tensorflow, was Keras.


During the training phase, we add a dropout layer after each of the convolutional layer as well
as fully connected layer. For keeping the consistency throughout all experimental models,
the dropout rate after each convolutional layer was chosen as 20% while it was 50% for all
fully connected layers.

We experimented with 3 convolutional layers & 4 hidden layers, 3 convolutional layers &
5 hidden layers, 4 convolutional layers & 4 hidden layers, 4 convolutional layers & 5 hidden
layers, 5 convolutional layers & 1 hidden layer and 5 convolutional layers & 2 hidden layers.
Note that, low number of convolutional layers increase the number of parameters in the
networks significantly while a high number of hidden layers introduce more parameters in the
Networks.

We trained ten different models for each of the network structures that we mentioned
above e.g. 3 convolutional layers & 4 hidden layers and so on, with the implementation of
dropout in each of the models. The loss function/objective function we deployed during the
model training was binary cross entropy since we are dealing with a two-class classification
problem.

In total, we had 260 microscopic images of blood samples. We trained our models with
200 images, while 30 images were kept for validation and the remaining 30 for the testing phase. 
Each of the models was trained with 50 epochs i.e. we passed the entire training set
through the Network 50 times. Each time the the set of parameters were different since 
we randomly dropping units from the networks.

During test time, we passed the test data 50 times through the network by keeping the dropout 
active and take the average in the end. This is also known as MC dropout [Gal and Ghahramani, 2015].

## 2. Files:


a) Model_combined.R file contains all trained models in six different network structures.The network structures we experimented with are 3 Convolutional Layers + 4 Hidden Layers, 3 Convolutional Layers + 5 Hidden Layers, 4 Convolutional Layers + 4 Hidden Layers, 4 Convolutional Layers + 5 Hidden Layers, 5 Convolutional Layers + 1 Hidden Layers, 5 Convolutional Layers + 2 Hidden Layers.

b) Visualization_combined.R contains the visualization of errors produced by models during test time, predictive uncertainty  and checking for overfittingness of the models of six network structures.

c) Loading_images.R is for how I load the image data into directories. The file path will depend of your local directory.

d) Images folder contains some of the images from our experiments.

