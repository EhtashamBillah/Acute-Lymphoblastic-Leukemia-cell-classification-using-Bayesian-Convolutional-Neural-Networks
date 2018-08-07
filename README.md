# Acute Lymphoblastic Leukemia cell classification from microscopic images using Bayesian Convolutional Neural Networks
In this project, we deploy the Bayesian Convolution Neural Networks (BCNN), proposed
by Gal and Ghahramani [2015] to classify microscopic images of blood samples (lymphocyte
cells). The data contains 260 microscopic images of cancerous and non-cancerous lymphocyte
cells. We experiment with different network structures to obtain the model that return
lowest error rate in classifying the images. We estimate the uncertainty for the predictions
made by the models which in turn can assist a doctor in better decision making. The
Stochastic Regularization Technique (SRT), popularly known as dropout is utilized in the
BCNN structure to obtain the Bayesian interpretation.
