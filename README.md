# Acute Lymphoblastic Leukemia cell classification from microscopic images using Bayesian Convolutional Neural Networks
In this project, we deploy the Bayesian Convolution Neural Networks (BCNN), proposed
by Gal and Ghahramani [2015] to classify microscopic images of blood samples (lymphocyte
cells). The data contains 260 microscopic images of cancerous and non-cancerous lymphocyte
cells. We experiment with different network structures to obtain the model that return
lowest error rate in classifying the images. We estimate the uncertainty for the predictions
made by the models which in turn can assist a doctor in better decision making. The
Stochastic Regularization Technique (SRT), popularly known as Dropout is utilized in the
BCNN structure to obtain the Bayesian interpretation.

During the training phase, we add a dropout layer after each of the convolutional layer as well
as fully connected layer. For keeping the consistency throughout all experimental models,
the dropout rate after each convolutional layer was chosen as 20% while it was 50% for all
fully connected layers.
