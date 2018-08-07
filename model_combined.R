# To get started with Keras, we need to install the Keras R package, 
# the core Keras library, and a backend tensor engine (such as TensorFlow).

#Step : 1 # Run install.packages("keras")  # to install the Keras R package
#Step : 2 # Run require(keras)
#Step : 3 # Run install_keras()  # installs the core Keras library, and tensorflow

require(keras)
require(ggplot2)


##################################################################
# 3 CONV LAYER WITH 5 hidden LAYER (1st)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%                                        # training=TRUE keeps the dropout active during test time                                     
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   # means 1st hidden layer with 1024 units
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%  
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%  
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_3c_5h_1t <- keras_model(input,output)
summary(model_do_3c_5h_1t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_3c_5h_1t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)



validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)


#fitting the model using a batch generator
history_do_3c_5h_1t <- model_do_3c_5h_1t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


##################################################################
# 3 CONV LAYER WITH 5 hidden LAYER (2nd)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%  
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_3c_5h_2n <- keras_model(input,output)
summary(model_do_3c_5h_2n) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_3c_5h_2n %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)



validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)



#fitting the model using a batch generator
history_do_3c_5h_2n <- model_do_3c_5h_2n %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


##################################################################
# 3 CONV LAYER WITH 5 hidden LAYER (3rd)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_3c_5h_3r <- keras_model(input,output)
summary(model_do_3c_5h_3r) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_3c_5h_3r %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)




validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)



#fitting the model using a batch generator
history_do_3c_5h_3r <- model_do_3c_5h_3r %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


##################################################################
# 3 CONV LAYER WITH 5 hidden LAYER (4th)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_3c_5h_4t <- keras_model(input,output)
summary(model_do_3c_5h_4t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_3c_5h_4t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)




validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)



#fitting the model using a batch generator
history_do_3c_5h_4t <- model_do_3c_5h_4t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


##################################################################
# 3 CONV LAYER WITH 5 hidden LAYER (5th)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_3c_5h_5t <- keras_model(input,output)
summary(model_do_3c_5h_5t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_3c_5h_5t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)




validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)



#fitting the model using a batch generator
history_do_3c_5h_5t <- model_do_3c_5h_5t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


##################################################################
# 3 CONV LAYER WITH 5 hidden LAYER (6th)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_3c_5h_6t <- keras_model(input,output)
summary(model_do_3c_5h_6t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_3c_5h_6t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)




validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)



#fitting the model using a batch generator
history_do_3c_5h_6t <- model_do_3c_5h_6t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


##################################################################
# 3 CONV LAYER WITH 5 hidden LAYER (7th)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_3c_5h_7t <- keras_model(input,output)
summary(model_do_3c_5h_7t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_3c_5h_7t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)




validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)



#fitting the model using a batch generator
history_do_3c_5h_7t <- model_do_3c_5h_7t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


##################################################################
# 3 CONV LAYER WITH 5 hidden LAYER (8th)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_3c_5h_8t <- keras_model(input,output)
summary(model_do_3c_5h_8t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_3c_5h_8t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)




validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)



#fitting the model using a batch generator
history_do_3c_5h_8t <- model_do_3c_5h_8t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


##################################################################
# 3 CONV LAYER WITH 5 hidden LAYER (9th)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_3c_5h_9t <- keras_model(input,output)
summary(model_do_3c_5h_9t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_3c_5h_9t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)




validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)



#fitting the model using a batch generator
history_do_3c_5h_9t <- model_do_3c_5h_9t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


##################################################################
# 3 CONV LAYER WITH 5 hidden LAYER (10th)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_3c_5h_10t <- keras_model(input,output)
summary(model_do_3c_5h_10t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_3c_5h_10t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)




validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)



#fitting the model using a batch generator
history_do_3c_5h_10t <- model_do_3c_5h_10t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


#evaluate the model on test data (estimates the loss and accuracy in total)
test_generator <- flow_images_from_directory(
  test_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 30,
  class_mode = "binary",
  shuffle = FALSE
  
)

batch_test <- generator_next(test_generator)
str(batch_test)




# MC PREDICTION & EVALUATION
nsim <- 50
MC_Samples <- 1:50
mc_evalu_3con_5hid_1t <- list()
mc_evalu_3con_5hid_2n <- list()
mc_evalu_3con_5hid_3r <- list()
mc_evalu_3con_5hid_4t <- list()
mc_evalu_3con_5hid_5t <- list()
mc_evalu_3con_5hid_6t <- list()
mc_evalu_3con_5hid_7t <- list()
mc_evalu_3con_5hid_8t <- list()
mc_evalu_3con_5hid_9t <- list()
mc_evalu_3con_5hid_10t <- list()

mc_pred_3con_5hid_1t <- matrix(nrow = 30, ncol = 50)
mc_pred_3con_5hid_2n <- matrix(nrow = 30, ncol = 50)
mc_pred_3con_5hid_3r <- matrix(nrow = 30, ncol = 50)
mc_pred_3con_5hid_4t <- matrix(nrow = 30, ncol = 50)
mc_pred_3con_5hid_5t <- matrix(nrow = 30, ncol = 50)
mc_pred_3con_5hid_6t <- matrix(nrow = 30, ncol = 50)
mc_pred_3con_5hid_7t <- matrix(nrow = 30, ncol = 50)
mc_pred_3con_5hid_8t <- matrix(nrow = 30, ncol = 50)
mc_pred_3con_5hid_9t <- matrix(nrow = 30, ncol = 50)
mc_pred_3con_5hid_10t <- matrix(nrow = 30, ncol = 50)


for (i in 1:nsim){
  mc_evalu_3con_5hid_1t[[i]] <- model_do_3c_5h_1t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_3con_5hid_2n[[i]] <- model_do_3c_5h_2n %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_3con_5hid_3r[[i]] <- model_do_3c_5h_3r %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_3con_5hid_4t[[i]] <- model_do_3c_5h_4t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_3con_5hid_5t[[i]] <- model_do_3c_5h_5t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_3con_5hid_6t[[i]] <- model_do_3c_5h_6t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_3con_5hid_7t[[i]] <- model_do_3c_5h_7t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_3con_5hid_8t[[i]] <- model_do_3c_5h_8t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_3con_5hid_9t[[i]] <- model_do_3c_5h_9t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_3con_5hid_10t[[i]] <- model_do_3c_5h_10t %>% evaluate_generator(test_generator, steps = 1)
  mc_pred_3con_5hid_1t[,i] <- model_do_3c_5h_1t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_3con_5hid_2n[,i] <- model_do_3c_5h_2n %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_3con_5hid_3r[,i] <- model_do_3c_5h_3r %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_3con_5hid_4t[,i] <- model_do_3c_5h_4t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_3con_5hid_5t[,i] <- model_do_3c_5h_5t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_3con_5hid_6t[,i] <- model_do_3c_5h_6t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_3con_5hid_7t[,i] <- model_do_3c_5h_7t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_3con_5hid_8t[,i] <- model_do_3c_5h_8t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_3con_5hid_9t[,i] <- model_do_3c_5h_9t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_3con_5hid_10t[,i] <- model_do_3c_5h_10t %>% predict_generator(test_generator,steps=1,verbose=1)
}



mc_evalu_3con_5hid_1t <- data.frame(matrix(unlist(mc_evalu_3con_5hid_1t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_3con_5hid_2n <- data.frame(matrix(unlist(mc_evalu_3con_5hid_2n), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_3con_5hid_3r <- data.frame(matrix(unlist(mc_evalu_3con_5hid_3r), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_3con_5hid_4t <- data.frame(matrix(unlist(mc_evalu_3con_5hid_4t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_3con_5hid_5t <- data.frame(matrix(unlist(mc_evalu_3con_5hid_5t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_3con_5hid_6t <- data.frame(matrix(unlist(mc_evalu_3con_5hid_6t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_3con_5hid_7t <- data.frame(matrix(unlist(mc_evalu_3con_5hid_7t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_3con_5hid_8t <- data.frame(matrix(unlist(mc_evalu_3con_5hid_8t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_3con_5hid_9t <- data.frame(matrix(unlist(mc_evalu_3con_5hid_9t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_3con_5hid_10t <- data.frame(matrix(unlist(mc_evalu_3con_5hid_10t), nrow=nsim, byrow=T),stringsAsFactors=F)

colnames(mc_evalu_3con_5hid_1t) <- c("Loss","Accuracy")
colnames(mc_evalu_3con_5hid_2n) <- c("Loss","Accuracy")
colnames(mc_evalu_3con_5hid_3r) <- c("Loss","Accuracy")
colnames(mc_evalu_3con_5hid_4t) <- c("Loss","Accuracy")
colnames(mc_evalu_3con_5hid_5t) <- c("Loss","Accuracy")
colnames(mc_evalu_3con_5hid_6t) <- c("Loss","Accuracy")
colnames(mc_evalu_3con_5hid_7t) <- c("Loss","Accuracy")
colnames(mc_evalu_3con_5hid_8t) <- c("Loss","Accuracy")
colnames(mc_evalu_3con_5hid_9t) <- c("Loss","Accuracy")
colnames(mc_evalu_3con_5hid_10t) <- c("Loss","Accuracy")



# Estimating Accuracy
apply(mc_evalu_3con_5hid_1t,2,mean) 
apply(mc_evalu_3con_5hid_2n,2,mean) 
apply(mc_evalu_3con_5hid_3r,2,mean) 
apply(mc_evalu_3con_5hid_4t,2,mean) 
apply(mc_evalu_3con_5hid_5t,2,mean) 
apply(mc_evalu_3con_5hid_6t,2,mean) 
apply(mc_evalu_3con_5hid_7t,2,mean) 
apply(mc_evalu_3con_5hid_8t,2,mean) 
apply(mc_evalu_3con_5hid_9t,2,mean) 
apply(mc_evalu_3con_5hid_10t,2,mean) 



table_thesis_3con_5hid <- t(cbind(apply(mc_evalu_3con_5hid_1t,2,mean),
                                  apply(mc_evalu_3con_5hid_2n,2,mean),
                                  apply(mc_evalu_3con_5hid_3r,2,mean),
                                  apply(mc_evalu_3con_5hid_4t,2,mean),
                                  apply(mc_evalu_3con_5hid_5t,2,mean),
                                  apply(mc_evalu_3con_5hid_6t,2,mean),
                                  apply(mc_evalu_3con_5hid_7t,2,mean),
                                  apply(mc_evalu_3con_5hid_8t,2,mean),
                                  apply(mc_evalu_3con_5hid_9t,2,mean),
                                  apply(mc_evalu_3con_5hid_10t,2,mean)))

mean(table_thesis_3con_5hid[,2])
sd(table_thesis_3con_5hid[,2])




##################################################################
# 3 CONV LAYER WITH 4 hidden LAYER (1st)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_3c_4h_1t <- keras_model(input,output)
summary(model_do_3c_4h_1t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_3c_4h_1t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)




validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)



#fitting the model using a batch generator
history_do_3c_4h_1t <- model_do_3c_4h_1t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


##################################################################
# 3 CONV LAYER WITH 4 hidden LAYER (2nd)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_3c_4h_2n <- keras_model(input,output)
summary(model_do_3c_4h_2n) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_3c_4h_2n %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)




validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)



#fitting the model using a batch generator
history_do_3c_4h_2n <- model_do_3c_4h_2n %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


##################################################################
# 3 CONV LAYER WITH 4 hidden LAYER (3rd)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_3c_4h_3r <- keras_model(input,output)
summary(model_do_3c_4h_3r) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_3c_4h_3r %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)




validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)



#fitting the model using a batch generator
history_do_3c_4h_3r <- model_do_3c_4h_3r %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


##################################################################
# 3 CONV LAYER WITH 4 hidden LAYER (4th)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_3c_4h_4t <- keras_model(input,output)
summary(model_do_3c_4h_4t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_3c_4h_4t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)



validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)



#fitting the model using a batch generator
history_do_3c_4h_4t <- model_do_3c_4h_4t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


##################################################################
# 3 CONV LAYER WITH 4 hidden LAYER (5th)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_3c_4h_5t <- keras_model(input,output)
summary(model_do_3c_4h_5t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_3c_4h_5t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)




validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)



#fitting the model using a batch generator
history_do_3c_4h_5t <- model_do_3c_4h_5t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


##################################################################
# 3 CONV LAYER WITH 4 hidden LAYER (6th)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_3c_4h_6t <- keras_model(input,output)
summary(model_do_3c_4h_6t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_3c_4h_6t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)




validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)



#fitting the model using a batch generator
history_do_3c_4h_6t <- model_do_3c_4h_6t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


##################################################################
# 3 CONV LAYER WITH 4 hidden LAYER (7th)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_3c_4h_7t <- keras_model(input,output)
summary(model_do_3c_4h_7t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_3c_4h_7t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)



validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)



#fitting the model using a batch generator
history_do_3c_4h_7t <- model_do_3c_4h_7t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


##################################################################
# 3 CONV LAYER WITH 4 hidden LAYER (8th)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_3c_4h_8t <- keras_model(input,output)
summary(model_do_3c_4h_8t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_3c_4h_8t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)




validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)



#fitting the model using a batch generator
history_do_3c_4h_8t <- model_do_3c_4h_8t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


##################################################################
# 3 CONV LAYER WITH 4 hidden LAYER (9th)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_3c_4h_9t <- keras_model(input,output)
summary(model_do_3c_4h_9t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_3c_4h_9t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)




validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)



#fitting the model using a batch generator
history_do_3c_4h_9t <- model_do_3c_4h_9t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


##################################################################
# 3 CONV LAYER WITH 4 hidden LAYER (10th)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_3c_4h_10t <- keras_model(input,output)
summary(model_do_3c_4h_10t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_3c_4h_10t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)




validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)



#fitting the model using a batch generator
history_do_3c_4h_10t <- model_do_3c_4h_10t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)



#evaluate the model on test data (estimates the loss and accuracy in total)
test_generator <- flow_images_from_directory(
  test_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 30,
  class_mode = "binary",
  shuffle = FALSE
  
)

batch_test <- generator_next(test_generator)
str(batch_test)



# MC PREDICTION & EVALUATION
nsim <- 50
mc_evalu_3con_4hid_1t <- list()
mc_evalu_3con_4hid_2n <- list()
mc_evalu_3con_4hid_3r <- list()
mc_evalu_3con_4hid_4t <- list()
mc_evalu_3con_4hid_5t <- list()
mc_evalu_3con_4hid_6t <- list()
mc_evalu_3con_4hid_7t <- list()
mc_evalu_3con_4hid_8t <- list()
mc_evalu_3con_4hid_9t <- list()
mc_evalu_3con_4hid_10t <- list()

mc_pred_3con_4hid_1t <- matrix(nrow = 30, ncol = 50)
mc_pred_3con_4hid_2n <- matrix(nrow = 30, ncol = 50)
mc_pred_3con_4hid_3r <- matrix(nrow = 30, ncol = 50)
mc_pred_3con_4hid_4t <- matrix(nrow = 30, ncol = 50)
mc_pred_3con_4hid_5t <- matrix(nrow = 30, ncol = 50)
mc_pred_3con_4hid_6t <- matrix(nrow = 30, ncol = 50)
mc_pred_3con_4hid_7t <- matrix(nrow = 30, ncol = 50)
mc_pred_3con_4hid_8t <- matrix(nrow = 30, ncol = 50)
mc_pred_3con_4hid_9t <- matrix(nrow = 30, ncol = 50)
mc_pred_3con_4hid_10t <- matrix(nrow = 30, ncol = 50)


for (i in 1:nsim){
  mc_evalu_3con_4hid_1t[[i]] <- model_do_3c_4h_1t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_3con_4hid_2n[[i]] <- model_do_3c_4h_2n %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_3con_4hid_3r[[i]] <- model_do_3c_4h_3r %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_3con_4hid_4t[[i]] <- model_do_3c_4h_4t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_3con_4hid_5t[[i]] <- model_do_3c_4h_5t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_3con_4hid_6t[[i]] <- model_do_3c_4h_6t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_3con_4hid_7t[[i]] <- model_do_3c_4h_7t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_3con_4hid_8t[[i]] <- model_do_3c_4h_8t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_3con_4hid_9t[[i]] <- model_do_3c_4h_9t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_3con_4hid_10t[[i]] <- model_do_3c_4h_10t %>% evaluate_generator(test_generator, steps = 1)
  mc_pred_3con_4hid_1t[,i] <- model_do_3c_4h_1t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_3con_4hid_2n[,i] <- model_do_3c_4h_2n %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_3con_4hid_3r[,i] <- model_do_3c_4h_3r %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_3con_4hid_4t[,i] <- model_do_3c_4h_4t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_3con_4hid_5t[,i] <- model_do_3c_4h_5t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_3con_4hid_6t[,i] <- model_do_3c_4h_6t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_3con_4hid_7t[,i] <- model_do_3c_4h_7t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_3con_4hid_8t[,i] <- model_do_3c_4h_8t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_3con_4hid_9t[,i] <- model_do_3c_4h_9t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_3con_4hid_10t[,i] <- model_do_3c_4h_10t %>% predict_generator(test_generator,steps=1,verbose=1)
}

apply(mc_pred_3con_4hid_1t,1,mean)
plot(apply(mc_pred_3con_4hid_10t,1,mean))


# Estimating Accuracy
mc_evalu_3con_4hid_1t <- data.frame(matrix(unlist(mc_evalu_3con_4hid_1t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_3con_4hid_2n <- data.frame(matrix(unlist(mc_evalu_3con_4hid_2n), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_3con_4hid_3r <- data.frame(matrix(unlist(mc_evalu_3con_4hid_3r), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_3con_4hid_4t <- data.frame(matrix(unlist(mc_evalu_3con_4hid_4t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_3con_4hid_5t <- data.frame(matrix(unlist(mc_evalu_3con_4hid_5t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_3con_4hid_6t <- data.frame(matrix(unlist(mc_evalu_3con_4hid_6t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_3con_4hid_7t <- data.frame(matrix(unlist(mc_evalu_3con_4hid_7t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_3con_4hid_8t <- data.frame(matrix(unlist(mc_evalu_3con_4hid_8t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_3con_4hid_9t <- data.frame(matrix(unlist(mc_evalu_3con_4hid_9t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_3con_4hid_10t <- data.frame(matrix(unlist(mc_evalu_3con_4hid_10t), nrow=nsim, byrow=T),stringsAsFactors=F)

colnames(mc_evalu_3con_4hid_1t) <- c("Loss","Accuracy")
colnames(mc_evalu_3con_4hid_2n) <- c("Loss","Accuracy")
colnames(mc_evalu_3con_4hid_3r) <- c("Loss","Accuracy")
colnames(mc_evalu_3con_4hid_4t) <- c("Loss","Accuracy")
colnames(mc_evalu_3con_4hid_5t) <- c("Loss","Accuracy")
colnames(mc_evalu_3con_4hid_6t) <- c("Loss","Accuracy")
colnames(mc_evalu_3con_4hid_7t) <- c("Loss","Accuracy")
colnames(mc_evalu_3con_4hid_8t) <- c("Loss","Accuracy")
colnames(mc_evalu_3con_4hid_9t) <- c("Loss","Accuracy")
colnames(mc_evalu_3con_4hid_10t) <- c("Loss","Accuracy")



apply(mc_evalu_3con_4hid_1t,2,mean) 
apply(mc_evalu_3con_4hid_2n,2,mean) 
apply(mc_evalu_3con_4hid_3r,2,mean) 
apply(mc_evalu_3con_4hid_4t,2,mean) 
apply(mc_evalu_3con_4hid_5t,2,mean) 
apply(mc_evalu_3con_4hid_6t,2,mean) 
apply(mc_evalu_3con_4hid_7t,2,mean) 
apply(mc_evalu_3con_4hid_8t,2,mean) 
apply(mc_evalu_3con_4hid_9t,2,mean) 
apply(mc_evalu_3con_4hid_10t,2,mean) 



table_thesis_3con_4hid <- t(cbind(apply(mc_evalu_3con_4hid_1t,2,mean),
                                  apply(mc_evalu_3con_4hid_2n,2,mean),
                                  apply(mc_evalu_3con_4hid_3r,2,mean),
                                  apply(mc_evalu_3con_4hid_4t,2,mean),
                                  apply(mc_evalu_3con_4hid_5t,2,mean),
                                  apply(mc_evalu_3con_4hid_6t,2,mean),
                                  apply(mc_evalu_3con_4hid_7t,2,mean),
                                  apply(mc_evalu_3con_4hid_8t,2,mean),
                                  apply(mc_evalu_3con_4hid_9t,2,mean),
                                  apply(mc_evalu_3con_4hid_10t,2,mean)))

mean(table_thesis_3con_4hid[,2])
sd(table_thesis_3con_4hid[,2])




##################################################################
# 4 CONV LAYER WITH 5 hidden LAYER (1st)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   # means 1st hidden layer with 1024 units
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_4c_5h_1t <- keras_model(input,output)
summary(model_do_4c_5h_1t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution



#configuring the model for training
model_do_4c_5h_1t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)



validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)



#fitting the model using a batch generator
history_do_4c_5h_1t <- model_do_4c_5h_1t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)



##################################################################
# 4 CONV LAYER WITH 5 hidden LAYER

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_4c_5h_2n <- keras_model(input,output)
summary(model_do_4c_5h_2n) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_4c_5h_2n %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)




validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)



#fitting the model using a batch generator
history_do_4c_5h_2n <- model_do_4c_5h_2n %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


##################################################################
# 4 CONV LAYER WITH 5 hidden LAYER (3rd)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   # means 1st hidden layer with 1024 units
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_4c_5h_3r <- keras_model(input,output)
summary(model_do_4c_5h_3r) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_4c_5h_3r %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)



validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)



#fitting the model using a batch generator
history_do_4c_5h_3r <- model_do_4c_5h_3r %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)



##################################################################
# 4 CONV LAYER WITH 5 hidden LAYER (4th)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_4c_5h_4t <- keras_model(input,output)
summary(model_do_4c_5h_4t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_4c_5h_4t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)



validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)



#fitting the model using a batch generator
history_do_4c_5h_4t <- model_do_4c_5h_4t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)



##################################################################
# 4 CONV LAYER WITH 5 hidden LAYER (5th)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_4c_5h_5t <- keras_model(input,output)
summary(model_do_4c_5h_5t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_4c_5h_5t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)


validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)



#fitting the model using a batch generator
history_do_4c_5h_5t <- model_do_4c_5h_5t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)



##################################################################
# 4 CONV LAYER WITH 5 hidden LAYER (6th)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_4c_5h_6t <- keras_model(input,output)
summary(model_do_4c_5h_6t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_4c_5h_6t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)



validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)



#fitting the model using a batch generator
history_do_4c_5h_6t <- model_do_4c_5h_6t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)



##################################################################
# 4 CONV LAYER WITH 5 hidden LAYER (7th)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_4c_5h_7t <- keras_model(input,output)
summary(model_do_4c_5h_7t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_4c_5h_7t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)




validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)



#fitting the model using a batch generator
history_do_4c_5h_7t <- model_do_4c_5h_7t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)



##################################################################
# 4 CONV LAYER WITH 5 hidden LAYER (8th)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_4c_5h_8t <- keras_model(input,output)
summary(model_do_4c_5h_8t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_4c_5h_8t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)




validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)



#fitting the model using a batch generator
history_do_4c_5h_8t <- model_do_4c_5h_8t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)




##################################################################
# 4 CONV LAYER WITH 5 hidden LAYER (9th)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_4c_5h_9t <- keras_model(input,output)
summary(model_do_4c_5h_9t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_4c_5h_9t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)




validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)



#fitting the model using a batch generator
history_do_4c_5h_9t <- model_do_4c_5h_9t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)



##################################################################
# 4 CONV LAYER WITH 5 hidden LAYER (10th)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_4c_5h_10t <- keras_model(input,output)
summary(model_do_4c_5h_10t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_4c_5h_10t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)




validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)



#fitting the model using a batch generator
history_do_4c_5h_10t <- model_do_4c_5h_10t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)

#evaluate the model on test data (estimates the loss and accuracy in total)
test_generator <- flow_images_from_directory(
  test_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 30,
  class_mode = "binary",
  shuffle = FALSE
  
)

batch_test <- generator_next(test_generator)
str(batch_test)




# MC PREDICTION & EVALUATION
nsim <- 50
mc_evalu_4con_5hid_1t <- list()
mc_evalu_4con_5hid_2n <- list()
mc_evalu_4con_5hid_3r <- list()
mc_evalu_4con_5hid_4t <- list()
mc_evalu_4con_5hid_5t <- list()
mc_evalu_4con_5hid_6t <- list()
mc_evalu_4con_5hid_7t <- list()
mc_evalu_4con_5hid_8t <- list()
mc_evalu_4con_5hid_9t <- list()
mc_evalu_4con_5hid_10t <- list()

mc_pred_4con_5hid_1t <- matrix(nrow = 30, ncol = 50)
mc_pred_4con_5hid_2n <- matrix(nrow = 30, ncol = 50)
mc_pred_4con_5hid_3r <- matrix(nrow = 30, ncol = 50)
mc_pred_4con_5hid_4t <- matrix(nrow = 30, ncol = 50)
mc_pred_4con_5hid_5t <- matrix(nrow = 30, ncol = 50)
mc_pred_4con_5hid_6t <- matrix(nrow = 30, ncol = 50)
mc_pred_4con_5hid_7t <- matrix(nrow = 30, ncol = 50)
mc_pred_4con_5hid_8t <- matrix(nrow = 30, ncol = 50)
mc_pred_4con_5hid_9t <- matrix(nrow = 30, ncol = 50)
mc_pred_4con_5hid_10t <- matrix(nrow = 30, ncol = 50)


for (i in 1:nsim){
  mc_evalu_4con_5hid_1t[[i]] <- model_do_4c_5h_1t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_4con_5hid_2n[[i]] <- model_do_4c_5h_2n %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_4con_5hid_3r[[i]] <- model_do_4c_5h_3r %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_4con_5hid_4t[[i]] <- model_do_4c_5h_4t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_4con_5hid_5t[[i]] <- model_do_4c_5h_5t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_4con_5hid_6t[[i]] <- model_do_4c_5h_6t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_4con_5hid_7t[[i]] <- model_do_4c_5h_7t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_4con_5hid_8t[[i]] <- model_do_4c_5h_8t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_4con_5hid_9t[[i]] <- model_do_4c_5h_9t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_4con_5hid_10t[[i]] <- model_do_4c_5h_10t %>% evaluate_generator(test_generator, steps = 1)
  mc_pred_4con_5hid_1t[,i] <- model_do_4c_5h_1t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_4con_5hid_2n[,i] <- model_do_4c_5h_2n %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_4con_5hid_3r[,i] <- model_do_4c_5h_3r %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_4con_5hid_4t[,i] <- model_do_4c_5h_4t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_4con_5hid_5t[,i] <- model_do_4c_5h_5t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_4con_5hid_6t[,i] <- model_do_4c_5h_6t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_4con_5hid_7t[,i] <- model_do_4c_5h_7t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_4con_5hid_8t[,i] <- model_do_4c_5h_8t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_4con_5hid_9t[,i] <- model_do_4c_5h_9t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_4con_5hid_10t[,i] <- model_do_4c_5h_10t %>% predict_generator(test_generator,steps=1,verbose=1)
}

apply(mc_pred_4con_5hid_1t,1,mean)
plot(apply(mc_pred_4con_5hid_10t,1,mean))


# Estimating Accuracy
mc_evalu_4con_5hid_1t <- data.frame(matrix(unlist(mc_evalu_4con_5hid_1t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_4con_5hid_2n <- data.frame(matrix(unlist(mc_evalu_4con_5hid_2n), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_4con_5hid_3r <- data.frame(matrix(unlist(mc_evalu_4con_5hid_3r), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_4con_5hid_4t <- data.frame(matrix(unlist(mc_evalu_4con_5hid_4t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_4con_5hid_5t <- data.frame(matrix(unlist(mc_evalu_4con_5hid_5t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_4con_5hid_6t <- data.frame(matrix(unlist(mc_evalu_4con_5hid_6t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_4con_5hid_7t <- data.frame(matrix(unlist(mc_evalu_4con_5hid_7t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_4con_5hid_8t <- data.frame(matrix(unlist(mc_evalu_4con_5hid_8t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_4con_5hid_9t <- data.frame(matrix(unlist(mc_evalu_4con_5hid_9t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_4con_5hid_10t <- data.frame(matrix(unlist(mc_evalu_4con_5hid_10t), nrow=nsim, byrow=T),stringsAsFactors=F)

colnames(mc_evalu_4con_5hid_1t) <- c("Loss","Accuracy")
colnames(mc_evalu_4con_5hid_2n) <- c("Loss","Accuracy")
colnames(mc_evalu_4con_5hid_3r) <- c("Loss","Accuracy")
colnames(mc_evalu_4con_5hid_4t) <- c("Loss","Accuracy")
colnames(mc_evalu_4con_5hid_5t) <- c("Loss","Accuracy")
colnames(mc_evalu_4con_5hid_6t) <- c("Loss","Accuracy")
colnames(mc_evalu_4con_5hid_7t) <- c("Loss","Accuracy")
colnames(mc_evalu_4con_5hid_8t) <- c("Loss","Accuracy")
colnames(mc_evalu_4con_5hid_9t) <- c("Loss","Accuracy")
colnames(mc_evalu_4con_5hid_10t) <- c("Loss","Accuracy")



apply(mc_evalu_4con_5hid_1t,2,mean) 
apply(mc_evalu_4con_5hid_2n,2,mean) 
apply(mc_evalu_4con_5hid_3r,2,mean) 
apply(mc_evalu_4con_5hid_4t,2,mean) 
apply(mc_evalu_4con_5hid_5t,2,mean) 
apply(mc_evalu_4con_5hid_6t,2,mean) 
apply(mc_evalu_4con_5hid_7t,2,mean) 
apply(mc_evalu_4con_5hid_8t,2,mean) 
apply(mc_evalu_4con_5hid_9t,2,mean) 
apply(mc_evalu_4con_5hid_10t,2,mean) 


table_thesis_4con_5hid <- t(cbind(apply(mc_evalu_4con_5hid_1t,2,mean),
                                  apply(mc_evalu_4con_5hid_2n,2,mean),
                                  apply(mc_evalu_4con_5hid_3r,2,mean),
                                  apply(mc_evalu_4con_5hid_4t,2,mean),
                                  apply(mc_evalu_4con_5hid_5t,2,mean),
                                  apply(mc_evalu_4con_5hid_6t,2,mean),
                                  apply(mc_evalu_4con_5hid_7t,2,mean),
                                  apply(mc_evalu_4con_5hid_8t,2,mean),
                                  apply(mc_evalu_4con_5hid_9t,2,mean),
                                  apply(mc_evalu_4con_5hid_10t,2,mean)))

mean(table_thesis_4con_5hid[,2])
sd(table_thesis_4con_5hid[,2])



##################################################################
# 4 CONV LAYER WITH 4 hidden LAYER (1st)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_4c_4h_1t <- keras_model(input,output)
summary(model_do_4c_4h_1t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_4c_4h_1t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)





validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)




#fitting the model using a batch generator
history_do_4c_4h_1t <- model_do_4c_4h_1t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)



##################################################################
# 4 CONV LAYER WITH 4 hidden LAYER

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_4c_4h_2n <- keras_model(input,output)
summary(model_do_4c_4h_2n) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_4c_4h_2n %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)





validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)




#fitting the model using a batch generator
history_do_4c_4h_2n <- model_do_4c_4h_2n %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


##################################################################
# 4 CONV LAYER WITH 4 hidden LAYER (3rd)
# total parameter 
###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_4c_4h_3r <- keras_model(input,output)
summary(model_do_4c_4h_3r) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_4c_4h_3r %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)





validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)




#fitting the model using a batch generator
history_do_4c_4h_3r <- model_do_4c_4h_3r %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)



##################################################################
# 4 CONV LAYER WITH 4 hidden LAYER (4th)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_4c_4h_4t <- keras_model(input,output)
summary(model_do_4c_4h_4t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_4c_4h_4t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)





validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)




#fitting the model using a batch generator
history_do_4c_4h_4t <- model_do_4c_4h_4t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)



##################################################################
# 4 CONV LAYER WITH 4 hidden LAYER (5th)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_4c_4h_5t <- keras_model(input,output)
summary(model_do_4c_4h_5t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_4c_4h_5t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)





validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)




#fitting the model using a batch generator
history_do_4c_4h_5t <- model_do_4c_4h_5t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)



##################################################################
# 4 CONV LAYER WITH 4 hidden LAYER (6th)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_4c_4h_6t <- keras_model(input,output)
summary(model_do_4c_4h_6t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_4c_4h_6t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)





validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)




#fitting the model using a batch generator
history_do_4c_4h_6t <- model_do_4c_4h_6t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)



##################################################################
# 4 CONV LAYER WITH 4 hidden LAYER (7th)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_4c_4h_7t <- keras_model(input,output)
summary(model_do_4c_4h_7t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_4c_4h_7t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)





validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)




#fitting the model using a batch generator
history_do_4c_4h_7t <- model_do_4c_4h_7t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)



##################################################################
# 4 CONV LAYER WITH 4 hidden LAYER (8th)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_4c_4h_8t <- keras_model(input,output)
summary(model_do_4c_4h_8t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_4c_4h_8t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)





validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)




#fitting the model using a batch generator
history_do_4c_4h_8t <- model_do_4c_4h_8t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)




##################################################################
# 4 CONV LAYER WITH 4 hidden LAYER (9th)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_4c_4h_9t <- keras_model(input,output)
summary(model_do_4c_4h_9t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_4c_4h_9t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)





validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)




#fitting the model using a batch generator
history_do_4c_4h_9t <- model_do_4c_4h_9t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)



##################################################################
# 4 CONV LAYER WITH 4 hidden LAYER (10th)

###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_4c_4h_10t <- keras_model(input,output)
summary(model_do_4c_4h_10t) 

# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution


#configuring the model for training
model_do_4c_4h_10t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)


#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)



validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)



#fitting the model using a batch generator
history_do_4c_4h_10t <- model_do_4c_4h_10t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


#evaluate the model on test data (estimates the loss and accuracy in total)
test_generator <- flow_images_from_directory(
  test_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 30,
  class_mode = "binary",
  shuffle = FALSE
  
)

batch_test <- generator_next(test_generator)
str(batch_test)



# MC PREDICTION & EVALUATION
nsim <- 50
mc_evalu_4con_4hid_1t <- list()
mc_evalu_4con_4hid_2n <- list()
mc_evalu_4con_4hid_3r <- list()
mc_evalu_4con_4hid_4t <- list()
mc_evalu_4con_4hid_5t <- list()
mc_evalu_4con_4hid_6t <- list()
mc_evalu_4con_4hid_7t <- list()
mc_evalu_4con_4hid_8t <- list()
mc_evalu_4con_4hid_9t <- list()
mc_evalu_4con_4hid_10t <- list()

mc_pred_4con_4hid_1t <- matrix(nrow = 30, ncol = 50)
mc_pred_4con_4hid_2n <- matrix(nrow = 30, ncol = 50)
mc_pred_4con_4hid_3r <- matrix(nrow = 30, ncol = 50)
mc_pred_4con_4hid_4t <- matrix(nrow = 30, ncol = 50)
mc_pred_4con_4hid_5t <- matrix(nrow = 30, ncol = 50)
mc_pred_4con_4hid_6t <- matrix(nrow = 30, ncol = 50)
mc_pred_4con_4hid_7t <- matrix(nrow = 30, ncol = 50)
mc_pred_4con_4hid_8t <- matrix(nrow = 30, ncol = 50)
mc_pred_4con_4hid_9t <- matrix(nrow = 30, ncol = 50)
mc_pred_4con_4hid_10t <- matrix(nrow = 30, ncol = 50)


for (i in 1:nsim){
  mc_evalu_4con_4hid_1t[[i]] <- model_do_4c_4h_1t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_4con_4hid_2n[[i]] <- model_do_4c_4h_2n %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_4con_4hid_3r[[i]] <- model_do_4c_4h_3r %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_4con_4hid_4t[[i]] <- model_do_4c_4h_4t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_4con_4hid_5t[[i]] <- model_do_4c_4h_5t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_4con_4hid_6t[[i]] <- model_do_4c_4h_6t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_4con_4hid_7t[[i]] <- model_do_4c_4h_7t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_4con_4hid_8t[[i]] <- model_do_4c_4h_8t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_4con_4hid_9t[[i]] <- model_do_4c_4h_9t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_4con_4hid_10t[[i]] <- model_do_4c_4h_10t %>% evaluate_generator(test_generator, steps = 1)
  mc_pred_4con_4hid_1t[,i] <- model_do_4c_4h_1t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_4con_4hid_2n[,i] <- model_do_4c_4h_2n %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_4con_4hid_3r[,i] <- model_do_4c_4h_3r %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_4con_4hid_4t[,i] <- model_do_4c_4h_4t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_4con_4hid_5t[,i] <- model_do_4c_4h_5t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_4con_4hid_6t[,i] <- model_do_4c_4h_6t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_4con_4hid_7t[,i] <- model_do_4c_4h_7t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_4con_4hid_8t[,i] <- model_do_4c_4h_8t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_4con_4hid_9t[,i] <- model_do_4c_4h_9t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_4con_4hid_10t[,i] <- model_do_4c_4h_10t %>% predict_generator(test_generator,steps=1,verbose=1)
}

apply(mc_pred_4con_4hid_1t,1,mean)
plot(apply(mc_pred_4con_4hid_10t,1,mean))


# Estimating Accuracy
mc_evalu_4con_4hid_1t <- data.frame(matrix(unlist(mc_evalu_4con_4hid_1t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_4con_4hid_2n <- data.frame(matrix(unlist(mc_evalu_4con_4hid_2n), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_4con_4hid_3r <- data.frame(matrix(unlist(mc_evalu_4con_4hid_3r), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_4con_4hid_4t <- data.frame(matrix(unlist(mc_evalu_4con_4hid_4t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_4con_4hid_5t <- data.frame(matrix(unlist(mc_evalu_4con_4hid_5t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_4con_4hid_6t <- data.frame(matrix(unlist(mc_evalu_4con_4hid_6t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_4con_4hid_7t <- data.frame(matrix(unlist(mc_evalu_4con_4hid_7t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_4con_4hid_8t <- data.frame(matrix(unlist(mc_evalu_4con_4hid_8t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_4con_4hid_9t <- data.frame(matrix(unlist(mc_evalu_4con_4hid_9t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_4con_4hid_10t <- data.frame(matrix(unlist(mc_evalu_4con_4hid_10t), nrow=nsim, byrow=T),stringsAsFactors=F)

colnames(mc_evalu_4con_4hid_1t) <- c("Loss","Accuracy")
colnames(mc_evalu_4con_4hid_2n) <- c("Loss","Accuracy")
colnames(mc_evalu_4con_4hid_3r) <- c("Loss","Accuracy")
colnames(mc_evalu_4con_4hid_4t) <- c("Loss","Accuracy")
colnames(mc_evalu_4con_4hid_5t) <- c("Loss","Accuracy")
colnames(mc_evalu_4con_4hid_6t) <- c("Loss","Accuracy")
colnames(mc_evalu_4con_4hid_7t) <- c("Loss","Accuracy")
colnames(mc_evalu_4con_4hid_8t) <- c("Loss","Accuracy")
colnames(mc_evalu_4con_4hid_9t) <- c("Loss","Accuracy")
colnames(mc_evalu_4con_4hid_10t) <- c("Loss","Accuracy")



apply(mc_evalu_4con_4hid_1t,2,mean) 
apply(mc_evalu_4con_4hid_2n,2,mean) 
apply(mc_evalu_4con_4hid_3r,2,mean) 
apply(mc_evalu_4con_4hid_4t,2,mean) 
apply(mc_evalu_4con_4hid_5t,2,mean)
apply(mc_evalu_4con_4hid_6t,2,mean) 
apply(mc_evalu_4con_4hid_7t,2,mean)
apply(mc_evalu_4con_4hid_8t,2,mean) 
apply(mc_evalu_4con_4hid_9t,2,mean) 
apply(mc_evalu_4con_4hid_10t,2,mean)



table_thesis_4con_4hid <- t(cbind(apply(mc_evalu_4con_4hid_1t,2,mean),
                                  apply(mc_evalu_4con_4hid_2n,2,mean),
                                  apply(mc_evalu_4con_4hid_3r,2,mean),
                                  apply(mc_evalu_4con_4hid_4t,2,mean),
                                  apply(mc_evalu_4con_4hid_5t,2,mean),
                                  apply(mc_evalu_4con_4hid_6t,2,mean),
                                  apply(mc_evalu_4con_4hid_7t,2,mean),
                                  apply(mc_evalu_4con_4hid_8t,2,mean),
                                  apply(mc_evalu_4con_4hid_9t,2,mean),
                                  apply(mc_evalu_4con_4hid_10t,2,mean)))

mean(table_thesis_4con_4hid[,2])
sd(table_thesis_4con_4hid[,2])



###################################################################
# 5 CONV LAYER WITH 1 hidden LAYER 
###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   # means 1st hidden layer with 1024 units
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_5c_1h_1t <- keras_model(input,output)
summary(model_do_5c_1h_1t) 


# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution



#configuring the model for training
model_do_5c_1h_1t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)

#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)





validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)




#fitting the model using a batch generator
history_do_5c_1h_1t <- model_do_5c_1h_1t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)

###################################################################
# 5 CONV LAYER WITH 1 hidden LAYER (2nd)
###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_5c_1h_2n <- keras_model(input,output)
summary(model_do_5c_1h_2n) 


# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution



#configuring the model for training
model_do_5c_1h_2n %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)

#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)





validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)




#fitting the model using a batch generator
history_do_5c_1h_2n <- model_do_5c_1h_2n %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


###################################################################
# 5 CONV LAYER WITH 1 hidden LAYER (3rd)
###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_5c_1h_3r <- keras_model(input,output)
summary(model_do_5c_1h_3r) 


# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution



#configuring the model for training
model_do_5c_1h_3r %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)

#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)





validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)




#fitting the model using a batch generator
history_do_5c_1h_3r <- model_do_5c_1h_3r %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


###################################################################
# 5 CONV LAYER WITH 1 hidden LAYER (4th)
###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_5c_1h_4t <- keras_model(input,output)
summary(model_do_5c_1h_4t) 


# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution



#configuring the model for training
model_do_5c_1h_4t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)

#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)





validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)




#fitting the model using a batch generator
history_do_5c_1h_4t <- model_do_5c_1h_4t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


###################################################################
# 5 CONV LAYER WITH 1 hidden LAYER (5th)
###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_5c_1h_5t <- keras_model(input,output)
summary(model_do_5c_1h_5t) 


# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution



#configuring the model for training
model_do_5c_1h_5t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)

#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)





validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)




#fitting the model using a batch generator
history_do_5c_1h_5t <- model_do_5c_1h_5t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


###################################################################
# 5 CONV LAYER WITH 1 hidden LAYER (6th)
###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_5c_1h_6t <- keras_model(input,output)
summary(model_do_5c_1h_6t) 


# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution



#configuring the model for training
model_do_5c_1h_6t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)

#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)





validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)




#fitting the model using a batch generator
history_do_5c_1h_6t <- model_do_5c_1h_6t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


###################################################################
# 5 CONV LAYER WITH 1 hidden LAYER (7th)
###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_5c_1h_7t <- keras_model(input,output)
summary(model_do_5c_1h_7t) 


# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution



#configuring the model for training
model_do_5c_1h_7t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)

#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)





validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)




#fitting the model using a batch generator
history_do_5c_1h_7t <- model_do_5c_1h_7t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


###################################################################
# 5 CONV LAYER WITH 1 hidden LAYER (8th)
###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_5c_1h_8t <- keras_model(input,output)
summary(model_do_5c_1h_8t) 


# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution



#configuring the model for training
model_do_5c_1h_8t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)

#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)





validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)




#fitting the model using a batch generator
history_do_5c_1h_8t <- model_do_5c_1h_8t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


###################################################################
# 5 CONV LAYER WITH 1 hidden LAYER (9th)
###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_5c_1h_9t <- keras_model(input,output)
summary(model_do_5c_1h_9t) 


# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution



#configuring the model for training
model_do_5c_1h_9t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)

#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)





validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)




#fitting the model using a batch generator
history_do_5c_1h_9t <- model_do_5c_1h_9t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


###################################################################
# 5 CONV LAYER WITH 1 hidden LAYER (10th)
###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_5c_1h_10t <- keras_model(input,output)
summary(model_do_5c_1h_10t) 


# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution



#configuring the model for training
model_do_5c_1h_10t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)

#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)





validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)




#fitting the model using a batch generator
history_do_5c_1h_10t <- model_do_5c_1h_10t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)




#evaluate the model on test data (estimates the loss and accuracy in total)
validation_datagen <- image_data_generator(1/255)
test_generator <- flow_images_from_directory(
  test_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary"
  
)

batch_test <- generator_next(test_generator)
str(batch_test)




#evaluate the model on test data (estimates the loss and accuracy in total)
test_generator <- flow_images_from_directory(
  test_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 30,
  class_mode = "binary",
  shuffle = FALSE
  
)

batch_test <- generator_next(test_generator)
str(batch_test)
batch_test



# MC PREDICTION & EVALUATION
nsim <- 50
mc_evalu_5con_1hid_1t <- list()
mc_evalu_5con_1hid_2n <- list()
mc_evalu_5con_1hid_3r <- list()
mc_evalu_5con_1hid_4t <- list()
mc_evalu_5con_1hid_5t <- list()
mc_evalu_5con_1hid_6t <- list()
mc_evalu_5con_1hid_7t <- list()
mc_evalu_5con_1hid_8t <- list()
mc_evalu_5con_1hid_9t <- list()
mc_evalu_5con_1hid_10t <- list()

mc_pred_5con_1hid_1t <- matrix(nrow = 30, ncol = 50)
mc_pred_5con_1hid_2n <- matrix(nrow = 30, ncol = 50)
mc_pred_5con_1hid_3r <- matrix(nrow = 30, ncol = 50)
mc_pred_5con_1hid_4t <- matrix(nrow = 30, ncol = 50)
mc_pred_5con_1hid_5t <- matrix(nrow = 30, ncol = 50)
mc_pred_5con_1hid_6t <- matrix(nrow = 30, ncol = 50)
mc_pred_5con_1hid_7t <- matrix(nrow = 30, ncol = 50)
mc_pred_5con_1hid_8t <- matrix(nrow = 30, ncol = 50)
mc_pred_5con_1hid_9t <- matrix(nrow = 30, ncol = 50)
mc_pred_5con_1hid_10t <- matrix(nrow = 30, ncol = 50)


for (i in 1:nsim){
  mc_evalu_5con_1hid_1t[[i]] <- model_do_5c_1h_1t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_5con_1hid_2n[[i]] <- model_do_5c_1h_2n %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_5con_1hid_3r[[i]] <- model_do_5c_1h_3r %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_5con_1hid_4t[[i]] <- model_do_5c_1h_4t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_5con_1hid_5t[[i]] <- model_do_5c_1h_5t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_5con_1hid_6t[[i]] <- model_do_5c_1h_6t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_5con_1hid_7t[[i]] <- model_do_5c_1h_7t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_5con_1hid_8t[[i]] <- model_do_5c_1h_8t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_5con_1hid_9t[[i]] <- model_do_5c_1h_9t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_5con_1hid_10t[[i]] <- model_do_5c_1h_10t %>% evaluate_generator(test_generator, steps = 1)
  mc_pred_5con_1hid_1t[,i] <- model_do_5c_1h_1t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_5con_1hid_2n[,i] <- model_do_5c_1h_2n %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_5con_1hid_3r[,i] <- model_do_5c_1h_3r %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_5con_1hid_4t[,i] <- model_do_5c_1h_4t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_5con_1hid_5t[,i] <- model_do_5c_1h_5t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_5con_1hid_6t[,i] <- model_do_5c_1h_6t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_5con_1hid_7t[,i] <- model_do_5c_1h_7t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_5con_1hid_8t[,i] <- model_do_5c_1h_8t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_5con_1hid_9t[,i] <- model_do_5c_1h_9t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_5con_1hid_10t[,i] <- model_do_5c_1h_10t %>% predict_generator(test_generator,steps=1,verbose=1)
}

apply(mc_pred_5con_1hid_1t,1,mean)
plot(apply(mc_pred_5con_1hid_10t,1,mean))


# Estimating Accuracy
mc_evalu_5con_1hid_1t <- data.frame(matrix(unlist(mc_evalu_5con_1hid_1t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_5con_1hid_2n <- data.frame(matrix(unlist(mc_evalu_5con_1hid_2n), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_5con_1hid_3r <- data.frame(matrix(unlist(mc_evalu_5con_1hid_3r), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_5con_1hid_4t <- data.frame(matrix(unlist(mc_evalu_5con_1hid_4t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_5con_1hid_5t <- data.frame(matrix(unlist(mc_evalu_5con_1hid_5t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_5con_1hid_6t <- data.frame(matrix(unlist(mc_evalu_5con_1hid_6t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_5con_1hid_7t <- data.frame(matrix(unlist(mc_evalu_5con_1hid_7t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_5con_1hid_8t <- data.frame(matrix(unlist(mc_evalu_5con_1hid_8t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_5con_1hid_9t <- data.frame(matrix(unlist(mc_evalu_5con_1hid_9t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_5con_1hid_10t <- data.frame(matrix(unlist(mc_evalu_5con_1hid_10t), nrow=nsim, byrow=T),stringsAsFactors=F)

colnames(mc_evalu_5con_1hid_1t) <- c("Loss","Accuracy")
colnames(mc_evalu_5con_1hid_2n) <- c("Loss","Accuracy")
colnames(mc_evalu_5con_1hid_3r) <- c("Loss","Accuracy")
colnames(mc_evalu_5con_1hid_4t) <- c("Loss","Accuracy")
colnames(mc_evalu_5con_1hid_5t) <- c("Loss","Accuracy")
colnames(mc_evalu_5con_1hid_6t) <- c("Loss","Accuracy")
colnames(mc_evalu_5con_1hid_7t) <- c("Loss","Accuracy")
colnames(mc_evalu_5con_1hid_8t) <- c("Loss","Accuracy")
colnames(mc_evalu_5con_1hid_9t) <- c("Loss","Accuracy")
colnames(mc_evalu_5con_1hid_10t) <- c("Loss","Accuracy")



apply(mc_evalu_5con_1hid_1t,2,mean) 
apply(mc_evalu_5con_1hid_2n,2,mean) 
apply(mc_evalu_5con_1hid_3r,2,mean) 
apply(mc_evalu_5con_1hid_4t,2,mean) 
apply(mc_evalu_5con_1hid_5t,2,mean) 
apply(mc_evalu_5con_1hid_6t,2,mean)
apply(mc_evalu_5con_1hid_7t,2,mean)
apply(mc_evalu_5con_1hid_8t,2,mean) 
apply(mc_evalu_5con_1hid_9t,2,mean) 
apply(mc_evalu_5con_1hid_10t,2,mean) 

table_thesis_5con_1hid <- t(cbind(apply(mc_evalu_5con_1hid_1t,2,mean),
                                  apply(mc_evalu_5con_1hid_2n,2,mean),
                                  apply(mc_evalu_5con_1hid_3r,2,mean),
                                  apply(mc_evalu_5con_1hid_4t,2,mean),
                                  apply(mc_evalu_5con_1hid_5t,2,mean),
                                  apply(mc_evalu_5con_1hid_6t,2,mean),
                                  apply(mc_evalu_5con_1hid_7t,2,mean),
                                  apply(mc_evalu_5con_1hid_8t,2,mean),
                                  apply(mc_evalu_5con_1hid_9t,2,mean),
                                  apply(mc_evalu_5con_1hid_10t,2,mean)))

mean(table_thesis_5con_1hid[,2])
sd(table_thesis_5con_1hid[,2])



###################################################################
# 5 CONV LAYER WITH 2 hidden LAYER (1ST)
###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_5c_2h_1t <- keras_model(input,output)
summary(model_do_5c_2h_1t) 


# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution



#configuring the model for training
model_do_5c_2h_1t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)

#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)





validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)




#fitting the model using a batch generator
history_do_5c_2h_1t <- model_do_5c_2h_1t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


###################################################################
# 5 CONV LAYER WITH 2 hidden LAYER (2nd)
###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_5c_2h_2n <- keras_model(input,output)
summary(model_do_5c_2h_2n) 


# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution



#configuring the model for training
model_do_5c_2h_2n %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)

#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)





validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)




#fitting the model using a batch generator
history_do_5c_2h_2n <- model_do_5c_2h_2n %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


###################################################################
# 5 CONV LAYER WITH 2 hidden LAYER (3rd)
###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_5c_2h_3r <- keras_model(input,output)
summary(model_do_5c_2h_3r) 


# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution



#configuring the model for training
model_do_5c_2h_3r %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)

#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)





validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)




#fitting the model using a batch generator
history_do_5c_2h_3r <- model_do_5c_2h_3r %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


###################################################################
# 5 CONV LAYER WITH 2 hidden LAYER (4th)
###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_5c_2h_4t <- keras_model(input,output)
summary(model_do_5c_2h_4t) 


# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution



#configuring the model for training
model_do_5c_2h_4t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)

#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)





validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)




#fitting the model using a batch generator
history_do_5c_2h_4t <- model_do_5c_2h_4t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


###################################################################
# 5 CONV LAYER WITH 2 hidden LAYER (5th)
###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_5c_2h_5t <- keras_model(input,output)
summary(model_do_5c_2h_5t) 


# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution



#configuring the model for training
model_do_5c_2h_5t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)

#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)





validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)




#fitting the model using a batch generator
history_do_5c_2h_5t <- model_do_5c_2h_5t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


###################################################################
# 5 CONV LAYER WITH 2 hidden LAYER (6th)
###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_5c_2h_6t <- keras_model(input,output)
summary(model_do_5c_2h_6t) 


# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution



#configuring the model for training
model_do_5c_2h_6t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)

#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)





validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)




#fitting the model using a batch generator
history_do_5c_2h_6t <- model_do_5c_2h_6t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


###################################################################
# 5 CONV LAYER WITH 2 hidden LAYER (7th)
###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_5c_2h_7t <- keras_model(input,output)
summary(model_do_5c_2h_7t) 


# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution



#configuring the model for training
model_do_5c_2h_7t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)

#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)





validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)




#fitting the model using a batch generator
history_do_5c_2h_7t <- model_do_5c_2h_7t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


###################################################################
# 5 CONV LAYER WITH 2 hidden LAYER (8th)
###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_5c_2h_8t <- keras_model(input,output)
summary(model_do_5c_2h_8t) 


# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution



#configuring the model for training
model_do_5c_2h_8t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)

#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)





validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)




#fitting the model using a batch generator
history_do_5c_2h_8t <- model_do_5c_2h_8t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


###################################################################
# 5 CONV LAYER WITH 2 hidden LAYER (9th)
###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_5c_2h_9t <- keras_model(input,output)
summary(model_do_5c_2h_9t) 


# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution



#configuring the model for training
model_do_5c_2h_9t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)

#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)





validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)




#fitting the model using a batch generator
history_do_5c_2h_9t <- model_do_5c_2h_9t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


###################################################################
# 5 CONV LAYER WITH 2 hidden LAYER (10th)
###################################################################
#installing a CNN
drop_out1 <- layer_dropout(rate=0.2)
drop_out2 <- layer_dropout(rate=0.5)

input <- layer_input(shape=c(257,257,3))
output <- input %>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation = "relu",   #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
                input_shape = c(257,257,3)) %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=64,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=128,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter=256,kernel_size = c(3,3),activation = "relu") %>%
  drop_out1(training=TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1024,activation = "relu") %>%   
  drop_out2(training=TRUE) %>%
  layer_dense(units=1,activation = "sigmoid")      # gives the probability in favour of yes/positive/1

model_do_5c_2h_10t <- keras_model(input,output)
summary(model_do_5c_2h_10t) 


# kernel_size=size of the patches extracted from inputs
# filter = the number of filter computed by convolution



#configuring the model for training
model_do_5c_2h_10t %>% compile(
  loss="binary_crossentropy",
  optimizer = optimizer_rmsprop(lr=1e-4),
  metrics = c("accuracy")
)

#data preprocessing

datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(257,257),
  batch_size = 20,
  class_mode = "binary",
  shuffle = TRUE 
)





validation_datagen <- image_data_generator(
  rescale = 1/255
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 10,
  class_mode = "binary",
  shuffle = TRUE
)




#fitting the model using a batch generator
history_do_5c_2h_10t <- model_do_5c_2h_10t %>% fit_generator(
  train_generator,
  steps_per_epoch = 10, # training observation/batch size = 200/20
  epochs = 50,
  
  validation_data = validation_generator,
  validation_steps = 3  # validation observation/batch size = 30/10
)


#evaluate the model on test data (estimates the loss and accuracy in total)
test_generator <- flow_images_from_directory(
  test_dir,
  validation_datagen,
  target_size = c(257,257),
  batch_size = 30,
  class_mode = "binary",
  shuffle = FALSE
  
)

batch_test <- generator_next(test_generator)
str(batch_test)




# MC PREDICTION & EVALUATION
nsim <- 50
mc_evalu_5con_2hid_1t <- list()
mc_evalu_5con_2hid_2n <- list()
mc_evalu_5con_2hid_3r <- list()
mc_evalu_5con_2hid_4t <- list()
mc_evalu_5con_2hid_5t <- list()
mc_evalu_5con_2hid_6t <- list()
mc_evalu_5con_2hid_7t <- list()
mc_evalu_5con_2hid_8t <- list()
mc_evalu_5con_2hid_9t <- list()
mc_evalu_5con_2hid_10t <- list()

mc_pred_5con_2hid_1t <- matrix(nrow = 30, ncol = 50)
mc_pred_5con_2hid_2n <- matrix(nrow = 30, ncol = 50)
mc_pred_5con_2hid_3r <- matrix(nrow = 30, ncol = 50)
mc_pred_5con_2hid_4t <- matrix(nrow = 30, ncol = 50)
mc_pred_5con_2hid_5t <- matrix(nrow = 30, ncol = 50)
mc_pred_5con_2hid_6t <- matrix(nrow = 30, ncol = 50)
mc_pred_5con_2hid_7t <- matrix(nrow = 30, ncol = 50)
mc_pred_5con_2hid_8t <- matrix(nrow = 30, ncol = 50)
mc_pred_5con_2hid_9t <- matrix(nrow = 30, ncol = 50)
mc_pred_5con_2hid_10t <- matrix(nrow = 30, ncol = 50)


for (i in 1:nsim){
  mc_evalu_5con_2hid_1t[[i]] <- model_do_5c_2h_1t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_5con_2hid_2n[[i]] <- model_do_5c_2h_2n %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_5con_2hid_3r[[i]] <- model_do_5c_2h_3r %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_5con_2hid_4t[[i]] <- model_do_5c_2h_4t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_5con_2hid_5t[[i]] <- model_do_5c_2h_5t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_5con_2hid_6t[[i]] <- model_do_5c_2h_6t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_5con_2hid_7t[[i]] <- model_do_5c_2h_7t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_5con_2hid_8t[[i]] <- model_do_5c_2h_8t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_5con_2hid_9t[[i]] <- model_do_5c_2h_9t %>% evaluate_generator(test_generator, steps = 1)
  mc_evalu_5con_2hid_10t[[i]] <- model_do_5c_2h_10t %>% evaluate_generator(test_generator, steps = 1)
  mc_pred_5con_2hid_1t[,i] <- model_do_5c_2h_1t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_5con_2hid_2n[,i] <- model_do_5c_2h_2n %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_5con_2hid_3r[,i] <- model_do_5c_2h_3r %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_5con_2hid_4t[,i] <- model_do_5c_2h_4t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_5con_2hid_5t[,i] <- model_do_5c_2h_5t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_5con_2hid_6t[,i] <- model_do_5c_2h_6t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_5con_2hid_7t[,i] <- model_do_5c_2h_7t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_5con_2hid_8t[,i] <- model_do_5c_2h_8t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_5con_2hid_9t[,i] <- model_do_5c_2h_9t %>% predict_generator(test_generator,steps=1,verbose=1)
  mc_pred_5con_2hid_10t[,i] <- model_do_5c_2h_10t %>% predict_generator(test_generator,steps=1,verbose=1)
}

apply(mc_pred_5con_2hid_1t,1,mean)
plot(apply(mc_pred_5con_2hid_10t,1,mean))


# Estimating Accuracy
mc_evalu_5con_2hid_1t <- data.frame(matrix(unlist(mc_evalu_5con_2hid_1t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_5con_2hid_2n <- data.frame(matrix(unlist(mc_evalu_5con_2hid_2n), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_5con_2hid_3r <- data.frame(matrix(unlist(mc_evalu_5con_2hid_3r), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_5con_2hid_4t <- data.frame(matrix(unlist(mc_evalu_5con_2hid_4t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_5con_2hid_5t <- data.frame(matrix(unlist(mc_evalu_5con_2hid_5t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_5con_2hid_6t <- data.frame(matrix(unlist(mc_evalu_5con_2hid_6t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_5con_2hid_7t <- data.frame(matrix(unlist(mc_evalu_5con_2hid_7t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_5con_2hid_8t <- data.frame(matrix(unlist(mc_evalu_5con_2hid_8t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_5con_2hid_9t <- data.frame(matrix(unlist(mc_evalu_5con_2hid_9t), nrow=nsim, byrow=T),stringsAsFactors=F)
mc_evalu_5con_2hid_10t <- data.frame(matrix(unlist(mc_evalu_5con_2hid_10t), nrow=nsim, byrow=T),stringsAsFactors=F)

colnames(mc_evalu_5con_2hid_1t) <- c("Loss","Accuracy")
colnames(mc_evalu_5con_2hid_2n) <- c("Loss","Accuracy")
colnames(mc_evalu_5con_2hid_3r) <- c("Loss","Accuracy")
colnames(mc_evalu_5con_2hid_4t) <- c("Loss","Accuracy")
colnames(mc_evalu_5con_2hid_5t) <- c("Loss","Accuracy")
colnames(mc_evalu_5con_2hid_6t) <- c("Loss","Accuracy")
colnames(mc_evalu_5con_2hid_7t) <- c("Loss","Accuracy")
colnames(mc_evalu_5con_2hid_8t) <- c("Loss","Accuracy")
colnames(mc_evalu_5con_2hid_9t) <- c("Loss","Accuracy")
colnames(mc_evalu_5con_2hid_10t) <- c("Loss","Accuracy")



apply(mc_evalu_5con_2hid_1t,2,mean) 
apply(mc_evalu_5con_2hid_2n,2,mean) 
apply(mc_evalu_5con_2hid_3r,2,mean) 
apply(mc_evalu_5con_2hid_4t,2,mean) 
apply(mc_evalu_5con_2hid_5t,2,mean) 
apply(mc_evalu_5con_2hid_6t,2,mean) 
apply(mc_evalu_5con_2hid_7t,2,mean) 
apply(mc_evalu_5con_2hid_8t,2,mean) 
apply(mc_evalu_5con_2hid_9t,2,mean) 
apply(mc_evalu_5con_2hid_10t,2,mean) 



table_thesis_5con_2hid <- t(cbind(apply(mc_evalu_5con_2hid_1t,2,mean),
                                  apply(mc_evalu_5con_2hid_2n,2,mean),
                                  apply(mc_evalu_5con_2hid_3r,2,mean),
                                  apply(mc_evalu_5con_2hid_4t,2,mean),
                                  apply(mc_evalu_5con_2hid_5t,2,mean),
                                  apply(mc_evalu_5con_2hid_6t,2,mean),
                                  apply(mc_evalu_5con_2hid_7t,2,mean),
                                  apply(mc_evalu_5con_2hid_8t,2,mean),
                                  apply(mc_evalu_5con_2hid_9t,2,mean),
                                  apply(mc_evalu_5con_2hid_10t,2,mean)))

mean(table_thesis_5con_2hid[,2])
sd(table_thesis_5con_2hid[,2])




###########################################
############Sensitivity and Specificity
##########################################


#################
#3con 5hid
################

#1st
y_actual <- unlist(batch_test[2])
mean_3c_5h_1t<- apply(mc_pred_3con_5hid_1t,1,mean)
y_pred_3c_5h_1t<-ifelse(mean_3c_5h_1t<0.5,0,1)

table(y_pred_3c_5h_1t,y_actual) 
sens_3c_5h_1t <- table(y_pred_3c_5h_1t,y_actual)[1,1]/(table(y_pred_3c_5h_1t,y_actual)[1,1]+
                                                         table(y_pred_3c_5h_1t,y_actual)[2,1])

spec_3c_5h_1t <- table(y_pred_3c_5h_1t,y_actual)[2,2]/(table(y_pred_3c_5h_1t,y_actual)[1,2]+
                                                         table(y_pred_3c_5h_1t,y_actual)[2,2])

#2nd

y_actual <- unlist(batch_test[2])
mean_3c_5h_2n<- apply(mc_pred_3con_5hid_2n,1,mean)
y_pred_3c_5h_2n<-ifelse(mean_3c_5h_2n<0.5,0,1)

table(y_pred_3c_5h_2n,y_actual) 
sens_3c_5h_2n <- table(y_pred_3c_5h_2n,y_actual)[1,1]/(table(y_pred_3c_5h_2n,y_actual)[1,1]+
                                                         table(y_pred_3c_5h_2n,y_actual)[2,1])

spec_3c_5h_2n <- table(y_pred_3c_5h_2n,y_actual)[2,2]/(table(y_pred_3c_5h_2n,y_actual)[1,2]+
                                                         table(y_pred_3c_5h_2n,y_actual)[2,2])

# 3r

y_actual <- unlist(batch_test[2])
mean_3c_5h_3r<- apply(mc_pred_3con_5hid_3r,1,mean)
y_pred_3c_5h_3r<-ifelse(mean_3c_5h_3r<0.5,0,1)

table(y_pred_3c_5h_3r,y_actual) 
sens_3c_5h_3r <- table(y_pred_3c_5h_3r,y_actual)[1,1]/(table(y_pred_3c_5h_3r,y_actual)[1,1]+
                                                         table(y_pred_3c_5h_3r,y_actual)[2,1])

spec_3c_5h_3r <- table(y_pred_3c_5h_3r,y_actual)[2,2]/(table(y_pred_3c_5h_3r,y_actual)[1,2]+
                                                         table(y_pred_3c_5h_3r,y_actual)[2,2])

# 4t

y_actual <- unlist(batch_test[2])
mean_3c_5h_4t<- apply(mc_pred_3con_5hid_4t,1,mean)
y_pred_3c_5h_4t<-ifelse(mean_3c_5h_4t<0.5,0,1)

table(y_pred_3c_5h_4t,y_actual) 
sens_3c_5h_4t <- table(y_pred_3c_5h_4t,y_actual)[1,1]/(table(y_pred_3c_5h_4t,y_actual)[1,1]+
                                                         table(y_pred_3c_5h_4t,y_actual)[2,1])

spec_3c_5h_4t <- table(y_pred_3c_5h_4t,y_actual)[2,2]/(table(y_pred_3c_5h_4t,y_actual)[1,2]+
                                                         table(y_pred_3c_5h_4t,y_actual)[2,2])

# 5t

y_actual <- unlist(batch_test[2])
mean_3c_5h_5t<- apply(mc_pred_3con_5hid_5t,1,mean)
y_pred_3c_5h_5t<-ifelse(mean_3c_5h_5t<0.5,0,1)

table(y_pred_3c_5h_5t,y_actual) 
sens_3c_5h_5t <- table(y_pred_3c_5h_5t,y_actual)[1,1]/(table(y_pred_3c_5h_5t,y_actual)[1,1]+
                                                         table(y_pred_3c_5h_5t,y_actual)[2,1])

spec_3c_5h_5t <- table(y_pred_3c_5h_5t,y_actual)[2,2]/(table(y_pred_3c_5h_5t,y_actual)[1,2]+
                                                         table(y_pred_3c_5h_5t,y_actual)[2,2])

# 6t

y_actual <- unlist(batch_test[2])
mean_3c_5h_6t<- apply(mc_pred_3con_5hid_6t,1,mean)
y_pred_3c_5h_6t<-ifelse(mean_3c_5h_6t<0.5,0,1)

table(y_pred_3c_5h_6t,y_actual) 
sens_3c_5h_6t <- table(y_pred_3c_5h_6t,y_actual)[1,1]/(table(y_pred_3c_5h_6t,y_actual)[1,1]+
                                                         table(y_pred_3c_5h_6t,y_actual)[2,1])

spec_3c_5h_6t <- table(y_pred_3c_5h_6t,y_actual)[2,2]/(table(y_pred_3c_5h_6t,y_actual)[1,2]+
                                                         table(y_pred_3c_5h_6t,y_actual)[2,2])

# 7t

y_actual <- unlist(batch_test[2])
mean_3c_5h_7t<- apply(mc_pred_3con_5hid_7t,1,mean)
y_pred_3c_5h_7t<-ifelse(mean_3c_5h_7t<0.5,0,1)

table(y_pred_3c_5h_7t,y_actual) 
sens_3c_5h_7t <- table(y_pred_3c_5h_7t,y_actual)[1,1]/(table(y_pred_3c_5h_7t,y_actual)[1,1]+
                                                         table(y_pred_3c_5h_7t,y_actual)[2,1])

spec_3c_5h_7t <- table(y_pred_3c_5h_7t,y_actual)[2,2]/(table(y_pred_3c_5h_7t,y_actual)[1,2]+
                                                         table(y_pred_3c_5h_7t,y_actual)[2,2])
# 8t

y_actual <- unlist(batch_test[2])
mean_3c_5h_8t<- apply(mc_pred_3con_5hid_8t,1,mean)
y_pred_3c_5h_8t<-ifelse(mean_3c_5h_8t<0.5,0,1)

table(y_pred_3c_5h_8t,y_actual) 
sens_3c_5h_8t <- table(y_pred_3c_5h_8t,y_actual)[1,1]/(table(y_pred_3c_5h_8t,y_actual)[1,1]+
                                                         table(y_pred_3c_5h_8t,y_actual)[2,1])

spec_3c_5h_8t <- table(y_pred_3c_5h_8t,y_actual)[2,2]/(table(y_pred_3c_5h_8t,y_actual)[1,2]+
                                                         table(y_pred_3c_5h_8t,y_actual)[2,2])

# 9t

y_actual <- unlist(batch_test[2])
mean_3c_5h_9t<- apply(mc_pred_3con_5hid_9t,1,mean)
y_pred_3c_5h_9t<-ifelse(mean_3c_5h_9t<0.5,0,1)

table(y_pred_3c_5h_9t,y_actual) 
sens_3c_5h_9t <- table(y_pred_3c_5h_9t,y_actual)[1,1]/(table(y_pred_3c_5h_9t,y_actual)[1,1]+
                                                         table(y_pred_3c_5h_9t,y_actual)[2,1])

spec_3c_5h_9t <- table(y_pred_3c_5h_9t,y_actual)[2,2]/(table(y_pred_3c_5h_9t,y_actual)[1,2]+
                                                         table(y_pred_3c_5h_9t,y_actual)[2,2])

# 10t

y_actual <- unlist(batch_test[2])
mean_3c_5h_10t<- apply(mc_pred_3con_5hid_10t,1,mean)
y_pred_3c_5h_10t<-ifelse(mean_3c_5h_10t<0.5,0,1)

table(y_pred_3c_5h_10t,y_actual) 
sens_3c_5h_10t <- table(y_pred_3c_5h_10t,y_actual)[1,1]/(table(y_pred_3c_5h_10t,y_actual)[1,1]+
                                                           table(y_pred_3c_5h_10t,y_actual)[2,1])

spec_3c_5h_10t <- table(y_pred_3c_5h_10t,y_actual)[2,2]/(table(y_pred_3c_5h_10t,y_actual)[1,2]+
                                                           table(y_pred_3c_5h_10t,y_actual)[2,2])



sens_spec_3c_5h_all <- cbind(rbind(sens_3c_5h_1t,sens_3c_5h_2n,sens_3c_5h_3r,
                                   sens_3c_5h_4t,sens_3c_5h_5t,sens_3c_5h_6t,
                                   sens_3c_5h_7t,sens_3c_5h_8t,sens_3c_5h_9t,
                                   sens_3c_5h_10t),
                             rbind(spec_3c_5h_1t,spec_3c_5h_2n,spec_3c_5h_3r,
                                   spec_3c_5h_4t,spec_3c_5h_5t,spec_3c_5h_6t,
                                   spec_3c_5h_7t,spec_3c_5h_8t,spec_3c_5h_9t,
                                   spec_3c_5h_10t))

colnames(sens_spec_3c_5h_all) <- c("sens","specs")
rownames(sens_spec_3c_5h_all) <- 1:10

sens_spec_3c_5h_all
apply(sens_spec_3c_5h_all,2,mean)
apply(sens_spec_3c_5h_all,2,sd)





#################
#3con 4hid
################

#1st
y_actual <- unlist(batch_test[2])
mean_3c_4h_1t<- apply(mc_pred_3con_4hid_1t,1,mean)
y_pred_3c_4h_1t<-ifelse(mean_3c_4h_1t<0.5,0,1)

table(y_pred_3c_4h_1t,y_actual) 
sens_3c_4h_1t <- table(y_pred_3c_4h_1t,y_actual)[1,1]/(table(y_pred_3c_4h_1t,y_actual)[1,1]+
                                                         table(y_pred_3c_4h_1t,y_actual)[2,1])

spec_3c_4h_1t <- table(y_pred_3c_4h_1t,y_actual)[2,2]/(table(y_pred_3c_4h_1t,y_actual)[1,2]+
                                                         table(y_pred_3c_4h_1t,y_actual)[2,2])

#2nd

y_actual <- unlist(batch_test[2])
mean_3c_4h_2n<- apply(mc_pred_3con_4hid_2n,1,mean)
y_pred_3c_4h_2n<-ifelse(mean_3c_4h_2n<0.5,0,1)

table(y_pred_3c_4h_2n,y_actual) 
sens_3c_4h_2n <- table(y_pred_3c_4h_2n,y_actual)[1,1]/(table(y_pred_3c_4h_2n,y_actual)[1,1]+
                                                         table(y_pred_3c_4h_2n,y_actual)[2,1])

spec_3c_4h_2n <- table(y_pred_3c_4h_2n,y_actual)[2,2]/(table(y_pred_3c_4h_2n,y_actual)[1,2]+
                                                         table(y_pred_3c_4h_2n,y_actual)[2,2])

# 3r

y_actual <- unlist(batch_test[2])
mean_3c_4h_3r<- apply(mc_pred_3con_4hid_3r,1,mean)
y_pred_3c_4h_3r<-ifelse(mean_3c_4h_3r<0.5,0,1)

table(y_pred_3c_4h_3r,y_actual) 
sens_3c_4h_3r <- table(y_pred_3c_4h_3r,y_actual)[1,1]/(table(y_pred_3c_4h_3r,y_actual)[1,1]+
                                                         table(y_pred_3c_4h_3r,y_actual)[2,1])

spec_3c_4h_3r <- table(y_pred_3c_4h_3r,y_actual)[2,2]/(table(y_pred_3c_4h_3r,y_actual)[1,2]+
                                                         table(y_pred_3c_4h_3r,y_actual)[2,2])

# 4t

y_actual <- unlist(batch_test[2])
mean_3c_4h_4t<- apply(mc_pred_3con_4hid_4t,1,mean)
y_pred_3c_4h_4t<-ifelse(mean_3c_4h_4t<0.5,0,1)

table(y_pred_3c_4h_4t,y_actual) 
sens_3c_4h_4t <- table(y_pred_3c_4h_4t,y_actual)[1,1]/(table(y_pred_3c_4h_4t,y_actual)[1,1]+
                                                         table(y_pred_3c_4h_4t,y_actual)[2,1])

spec_3c_4h_4t <- table(y_pred_3c_4h_4t,y_actual)[2,2]/(table(y_pred_3c_4h_4t,y_actual)[1,2]+
                                                         table(y_pred_3c_4h_4t,y_actual)[2,2])

# 5t

y_actual <- unlist(batch_test[2])
mean_3c_4h_5t<- apply(mc_pred_3con_4hid_5t,1,mean)
y_pred_3c_4h_5t<-ifelse(mean_3c_4h_5t<0.5,0,1)

table(y_pred_3c_4h_5t,y_actual) 
sens_3c_4h_5t <- table(y_pred_3c_4h_5t,y_actual)[1,1]/(table(y_pred_3c_4h_5t,y_actual)[1,1]+
                                                         table(y_pred_3c_4h_5t,y_actual)[2,1])

spec_3c_4h_5t <- table(y_pred_3c_4h_5t,y_actual)[2,2]/(table(y_pred_3c_4h_5t,y_actual)[1,2]+
                                                         table(y_pred_3c_4h_5t,y_actual)[2,2])

# 6t

y_actual <- unlist(batch_test[2])
mean_3c_4h_6t<- apply(mc_pred_3con_4hid_6t,1,mean)
y_pred_3c_4h_6t<-ifelse(mean_3c_4h_6t<0.5,0,1)

table(y_pred_3c_4h_6t,y_actual) 
sens_3c_4h_6t <- table(y_pred_3c_4h_6t,y_actual)[1,1]/(table(y_pred_3c_4h_6t,y_actual)[1,1]+
                                                         table(y_pred_3c_4h_6t,y_actual)[2,1])

spec_3c_4h_6t <- table(y_pred_3c_4h_6t,y_actual)[2,2]/(table(y_pred_3c_4h_6t,y_actual)[1,2]+
                                                         table(y_pred_3c_4h_6t,y_actual)[2,2])

# 7t

y_actual <- unlist(batch_test[2])
mean_3c_4h_7t<- apply(mc_pred_3con_4hid_7t,1,mean)
y_pred_3c_4h_7t<-ifelse(mean_3c_4h_7t<0.5,0,1)

table(y_pred_3c_4h_7t,y_actual) 
sens_3c_4h_7t <- table(y_pred_3c_4h_7t,y_actual)[1,1]/(table(y_pred_3c_4h_7t,y_actual)[1,1]+
                                                         table(y_pred_3c_4h_7t,y_actual)[2,1])

spec_3c_4h_7t <- table(y_pred_3c_4h_7t,y_actual)[2,2]/(table(y_pred_3c_4h_7t,y_actual)[1,2]+
                                                         table(y_pred_3c_4h_7t,y_actual)[2,2])
# 8t

y_actual <- unlist(batch_test[2])
mean_3c_4h_8t<- apply(mc_pred_3con_4hid_8t,1,mean)
y_pred_3c_4h_8t<-ifelse(mean_3c_4h_8t<0.5,0,1)

table(y_pred_3c_4h_8t,y_actual) 
sens_3c_4h_8t <- table(y_pred_3c_4h_8t,y_actual)[1,1]/(table(y_pred_3c_4h_8t,y_actual)[1,1]+
                                                         table(y_pred_3c_4h_8t,y_actual)[2,1])

spec_3c_4h_8t <- table(y_pred_3c_4h_8t,y_actual)[2,2]/(table(y_pred_3c_4h_8t,y_actual)[1,2]+
                                                         table(y_pred_3c_4h_8t,y_actual)[2,2])

# 9t

y_actual <- unlist(batch_test[2])
mean_3c_4h_9t<- apply(mc_pred_3con_4hid_9t,1,mean)
y_pred_3c_4h_9t<-ifelse(mean_3c_4h_9t<0.5,0,1)

table(y_pred_3c_4h_9t,y_actual) 
sens_3c_4h_9t <- table(y_pred_3c_4h_9t,y_actual)[1,1]/(table(y_pred_3c_4h_9t,y_actual)[1,1]+
                                                         table(y_pred_3c_4h_9t,y_actual)[2,1])

spec_3c_4h_9t <- table(y_pred_3c_4h_9t,y_actual)[2,2]/(table(y_pred_3c_4h_9t,y_actual)[1,2]+
                                                         table(y_pred_3c_4h_9t,y_actual)[2,2])

# 10t

y_actual <- unlist(batch_test[2])
mean_3c_4h_10t<- apply(mc_pred_3con_4hid_10t,1,mean)
y_pred_3c_4h_10t<-ifelse(mean_3c_4h_10t<0.5,0,1)

table(y_pred_3c_4h_10t,y_actual) 
sens_3c_4h_10t <- table(y_pred_3c_4h_10t,y_actual)[1,1]/(table(y_pred_3c_4h_10t,y_actual)[1,1]+
                                                           table(y_pred_3c_4h_10t,y_actual)[2,1])

spec_3c_4h_10t <- table(y_pred_3c_4h_10t,y_actual)[2,2]/(table(y_pred_3c_4h_10t,y_actual)[1,2]+
                                                           table(y_pred_3c_4h_10t,y_actual)[2,2])



sens_spec_3c_4h_all <- cbind(rbind(sens_3c_4h_1t,sens_3c_4h_2n,sens_3c_4h_3r,
                                   sens_3c_4h_4t,sens_3c_4h_5t,sens_3c_4h_6t,
                                   sens_3c_4h_7t,sens_3c_4h_8t,sens_3c_4h_9t,
                                   sens_3c_4h_10t),
                             rbind(spec_3c_4h_1t,spec_3c_4h_2n,spec_3c_4h_3r,
                                   spec_3c_4h_4t,spec_3c_4h_5t,spec_3c_4h_6t,
                                   spec_3c_4h_7t,spec_3c_4h_8t,spec_3c_4h_9t,
                                   spec_3c_4h_10t))

colnames(sens_spec_3c_4h_all) <- c("sens","specs")
rownames(sens_spec_3c_4h_all) <- 1:10

sens_spec_3c_4h_all
apply(sens_spec_3c_4h_all,2,mean)
apply(sens_spec_3c_4h_all,2,sd)



#################
# 4con 5hid
################


#1st
y_actual <- unlist(batch_test[2])
mean_4c_5h_1t<- apply(mc_pred_4con_5hid_1t,1,mean)
y_pred_4c_5h_1t<-ifelse(mean_4c_5h_1t<0.5,0,1)

table(y_pred_4c_5h_1t,y_actual) 
sens_4c_5h_1t <- table(y_pred_4c_5h_1t,y_actual)[1,1]/(table(y_pred_4c_5h_1t,y_actual)[1,1]+
                                                         table(y_pred_4c_5h_1t,y_actual)[2,1])

spec_4c_5h_1t <- table(y_pred_4c_5h_1t,y_actual)[2,2]/(table(y_pred_4c_5h_1t,y_actual)[1,2]+
                                                         table(y_pred_4c_5h_1t,y_actual)[2,2])

#2nd

y_actual <- unlist(batch_test[2])
mean_4c_5h_2n<- apply(mc_pred_4con_5hid_2n,1,mean)
y_pred_4c_5h_2n<-ifelse(mean_4c_5h_2n<0.5,0,1)

table(y_pred_4c_5h_2n,y_actual) 
sens_4c_5h_2n <- table(y_pred_4c_5h_2n,y_actual)[1,1]/(table(y_pred_4c_5h_2n,y_actual)[1,1]+
                                                         table(y_pred_4c_5h_2n,y_actual)[2,1])

spec_4c_5h_2n <- table(y_pred_4c_5h_2n,y_actual)[2,2]/(table(y_pred_4c_5h_2n,y_actual)[1,2]+
                                                         table(y_pred_4c_5h_2n,y_actual)[2,2])

# 3r

y_actual <- unlist(batch_test[2])
mean_4c_5h_3r<- apply(mc_pred_4con_5hid_3r,1,mean)
y_pred_4c_5h_3r<-ifelse(mean_4c_5h_3r<0.5,0,1)

table(y_pred_4c_5h_3r,y_actual) 
sens_4c_5h_3r <- table(y_pred_4c_5h_3r,y_actual)[1,1]/(table(y_pred_4c_5h_3r,y_actual)[1,1]+
                                                         table(y_pred_4c_5h_3r,y_actual)[2,1])

spec_4c_5h_3r <- table(y_pred_4c_5h_3r,y_actual)[2,2]/(table(y_pred_4c_5h_3r,y_actual)[1,2]+
                                                         table(y_pred_4c_5h_3r,y_actual)[2,2])

# 4t

y_actual <- unlist(batch_test[2])
mean_4c_5h_4t<- apply(mc_pred_4con_5hid_4t,1,mean)
y_pred_4c_5h_4t<-ifelse(mean_4c_5h_4t<0.5,0,1)

table(y_pred_4c_5h_4t,y_actual) 
sens_4c_5h_4t <- table(y_pred_4c_5h_4t,y_actual)[1,1]/(table(y_pred_4c_5h_4t,y_actual)[1,1]+
                                                         table(y_pred_4c_5h_4t,y_actual)[2,1])

spec_4c_5h_4t <- table(y_pred_4c_5h_4t,y_actual)[2,2]/(table(y_pred_4c_5h_4t,y_actual)[1,2]+
                                                         table(y_pred_4c_5h_4t,y_actual)[2,2])

# 5t

y_actual <- unlist(batch_test[2])
mean_4c_5h_5t<- apply(mc_pred_4con_5hid_5t,1,mean)
y_pred_4c_5h_5t<-ifelse(mean_4c_5h_5t<0.5,0,1)

table(y_pred_4c_5h_5t,y_actual) 
sens_4c_5h_5t <- table(y_pred_4c_5h_5t,y_actual)[1,1]/(table(y_pred_4c_5h_5t,y_actual)[1,1]+
                                                         table(y_pred_4c_5h_5t,y_actual)[2,1])

spec_4c_5h_5t <- table(y_pred_4c_5h_5t,y_actual)[2,2]/(table(y_pred_4c_5h_5t,y_actual)[1,2]+
                                                         table(y_pred_4c_5h_5t,y_actual)[2,2])

# 6t

y_actual <- unlist(batch_test[2])
mean_4c_5h_6t<- apply(mc_pred_4con_5hid_6t,1,mean)
y_pred_4c_5h_6t<-ifelse(mean_4c_5h_6t<0.5,0,1)

table(y_pred_4c_5h_6t,y_actual) 
sens_4c_5h_6t <- table(y_pred_4c_5h_6t,y_actual)[1,1]/(table(y_pred_4c_5h_6t,y_actual)[1,1]+
                                                         table(y_pred_4c_5h_6t,y_actual)[2,1])

spec_4c_5h_6t <- table(y_pred_4c_5h_6t,y_actual)[2,2]/(table(y_pred_4c_5h_6t,y_actual)[1,2]+
                                                         table(y_pred_4c_5h_6t,y_actual)[2,2])

# 7t

y_actual <- unlist(batch_test[2])
mean_4c_5h_7t<- apply(mc_pred_4con_5hid_7t,1,mean)
y_pred_4c_5h_7t<-ifelse(mean_4c_5h_7t<0.5,0,1)

table(y_pred_4c_5h_7t,y_actual) 
sens_4c_5h_7t <- table(y_pred_4c_5h_7t,y_actual)[1,1]/(table(y_pred_4c_5h_7t,y_actual)[1,1]+
                                                         table(y_pred_4c_5h_7t,y_actual)[2,1])

spec_4c_5h_7t <- table(y_pred_4c_5h_7t,y_actual)[2,2]/(table(y_pred_4c_5h_7t,y_actual)[1,2]+
                                                         table(y_pred_4c_5h_7t,y_actual)[2,2])
# 8t

y_actual <- unlist(batch_test[2])
mean_4c_5h_8t<- apply(mc_pred_4con_5hid_8t,1,mean)
y_pred_4c_5h_8t<-ifelse(mean_4c_5h_8t<0.5,0,1)

table(y_pred_4c_5h_8t,y_actual) 
sens_4c_5h_8t <- table(y_pred_4c_5h_8t,y_actual)[1,1]/(table(y_pred_4c_5h_8t,y_actual)[1,1]+
                                                         table(y_pred_4c_5h_8t,y_actual)[2,1])

spec_4c_5h_8t <- table(y_pred_4c_5h_8t,y_actual)[2,2]/(table(y_pred_4c_5h_8t,y_actual)[1,2]+
                                                         table(y_pred_4c_5h_8t,y_actual)[2,2])

# 9t

y_actual <- unlist(batch_test[2])
mean_4c_5h_9t<- apply(mc_pred_4con_5hid_9t,1,mean)
y_pred_4c_5h_9t<-ifelse(mean_4c_5h_9t<0.5,0,1)

table(y_pred_4c_5h_9t,y_actual) 
sens_4c_5h_9t <- table(y_pred_4c_5h_9t,y_actual)[1,1]/(table(y_pred_4c_5h_9t,y_actual)[1,1]+
                                                         table(y_pred_4c_5h_9t,y_actual)[2,1])

spec_4c_5h_9t <- table(y_pred_4c_5h_9t,y_actual)[2,2]/(table(y_pred_4c_5h_9t,y_actual)[1,2]+
                                                         table(y_pred_4c_5h_9t,y_actual)[2,2])

# 10t

y_actual <- unlist(batch_test[2])
mean_4c_5h_10t<- apply(mc_pred_4con_5hid_10t,1,mean)
y_pred_4c_5h_10t<-ifelse(mean_4c_5h_10t<0.5,0,1)

table(y_pred_4c_5h_10t,y_actual) 
sens_4c_5h_10t <- table(y_pred_4c_5h_10t,y_actual)[1,1]/(table(y_pred_4c_5h_10t,y_actual)[1,1]+
                                                           table(y_pred_4c_5h_10t,y_actual)[2,1])

spec_4c_5h_10t <- table(y_pred_4c_5h_10t,y_actual)[2,2]/(table(y_pred_4c_5h_10t,y_actual)[1,2]+
                                                           table(y_pred_4c_5h_10t,y_actual)[2,2])


sens_spec_4c_5h_all <- cbind(rbind(sens_4c_5h_1t,sens_4c_5h_2n,sens_4c_5h_3r,
                                   sens_4c_5h_4t,sens_4c_5h_5t,sens_4c_5h_6t,
                                   sens_4c_5h_7t,sens_4c_5h_8t,sens_4c_5h_9t,
                                   sens_4c_5h_10t),
                             rbind(spec_4c_5h_1t,spec_4c_5h_2n,spec_4c_5h_3r,
                                   spec_4c_5h_4t,spec_4c_5h_5t,spec_4c_5h_6t,
                                   spec_4c_5h_7t,spec_4c_5h_8t,spec_4c_5h_9t,
                                   spec_4c_5h_10t))

colnames(sens_spec_4c_5h_all) <- c("sens","specs")
rownames(sens_spec_4c_5h_all) <- 1:10

sens_spec_4c_5h_all
apply(sens_spec_4c_5h_all,2,mean)
apply(sens_spec_4c_5h_all,2,sd)

apply(sens_spec_3c_5h_all,2,mean)
apply(sens_spec_3c_5h_all,2,sd)




#################
#4con 4hid
################

#1st
y_actual <- unlist(batch_test[2])
mean_4c_4h_1t<- apply(mc_pred_4con_4hid_1t,1,mean)
y_pred_4c_4h_1t<-ifelse(mean_4c_4h_1t<0.5,0,1)

table(y_pred_4c_4h_1t,y_actual) 
sens_4c_4h_1t <- table(y_pred_4c_4h_1t,y_actual)[1,1]/(table(y_pred_4c_4h_1t,y_actual)[1,1]+
                                                         table(y_pred_4c_4h_1t,y_actual)[2,1])

spec_4c_4h_1t <- table(y_pred_4c_4h_1t,y_actual)[2,2]/(table(y_pred_4c_4h_1t,y_actual)[1,2]+
                                                         table(y_pred_4c_4h_1t,y_actual)[2,2])

#2nd

y_actual <- unlist(batch_test[2])
mean_4c_4h_2n<- apply(mc_pred_4con_4hid_2n,1,mean)
y_pred_4c_4h_2n<-ifelse(mean_4c_4h_2n<0.5,0,1)

table(y_pred_4c_4h_2n,y_actual) 
sens_4c_4h_2n <- table(y_pred_4c_4h_2n,y_actual)[1,1]/(table(y_pred_4c_4h_2n,y_actual)[1,1]+
                                                         table(y_pred_4c_4h_2n,y_actual)[2,1])

spec_4c_4h_2n <- table(y_pred_4c_4h_2n,y_actual)[2,2]/(table(y_pred_4c_4h_2n,y_actual)[1,2]+
                                                         table(y_pred_4c_4h_2n,y_actual)[2,2])

# 3r

y_actual <- unlist(batch_test[2])
mean_4c_4h_3r<- apply(mc_pred_4con_4hid_3r,1,mean)
y_pred_4c_4h_3r<-ifelse(mean_4c_4h_3r<0.5,0,1)

table(y_pred_4c_4h_3r,y_actual) 
sens_4c_4h_3r <- table(y_pred_4c_4h_3r,y_actual)[1,1]/(table(y_pred_4c_4h_3r,y_actual)[1,1]+
                                                         table(y_pred_4c_4h_3r,y_actual)[2,1])

spec_4c_4h_3r <- table(y_pred_4c_4h_3r,y_actual)[2,2]/(table(y_pred_4c_4h_3r,y_actual)[1,2]+
                                                         table(y_pred_4c_4h_3r,y_actual)[2,2])

# 4t

y_actual <- unlist(batch_test[2])
mean_4c_4h_4t<- apply(mc_pred_4con_4hid_4t,1,mean)
y_pred_4c_4h_4t<-ifelse(mean_4c_4h_4t<0.5,0,1)

table(y_pred_4c_4h_4t,y_actual) 
sens_4c_4h_4t <- table(y_pred_4c_4h_4t,y_actual)[1,1]/(table(y_pred_4c_4h_4t,y_actual)[1,1]+
                                                         table(y_pred_4c_4h_4t,y_actual)[2,1])

spec_4c_4h_4t <- table(y_pred_4c_4h_4t,y_actual)[2,2]/(table(y_pred_4c_4h_4t,y_actual)[1,2]+
                                                         table(y_pred_4c_4h_4t,y_actual)[2,2])

# 5t

y_actual <- unlist(batch_test[2])
mean_4c_4h_5t<- apply(mc_pred_4con_4hid_5t,1,mean)
y_pred_4c_4h_5t<-ifelse(mean_4c_4h_5t<0.5,0,1)

table(y_pred_4c_4h_5t,y_actual) 
sens_4c_4h_5t <- table(y_pred_4c_4h_5t,y_actual)[1,1]/(table(y_pred_4c_4h_5t,y_actual)[1,1]+
                                                         table(y_pred_4c_4h_5t,y_actual)[2,1])

spec_4c_4h_5t <- table(y_pred_4c_4h_5t,y_actual)[2,2]/(table(y_pred_4c_4h_5t,y_actual)[1,2]+
                                                         table(y_pred_4c_4h_5t,y_actual)[2,2])

# 6t

y_actual <- unlist(batch_test[2])
mean_4c_4h_6t<- apply(mc_pred_4con_4hid_6t,1,mean)
y_pred_4c_4h_6t<-ifelse(mean_4c_4h_6t<0.5,0,1)

table(y_pred_4c_4h_6t,y_actual) 
sens_4c_4h_6t <- table(y_pred_4c_4h_6t,y_actual)[1,1]/(table(y_pred_4c_4h_6t,y_actual)[1,1]+
                                                         table(y_pred_4c_4h_6t,y_actual)[2,1])

spec_4c_4h_6t <- table(y_pred_4c_4h_6t,y_actual)[2,2]/(table(y_pred_4c_4h_6t,y_actual)[1,2]+
                                                         table(y_pred_4c_4h_6t,y_actual)[2,2])

# 7t

y_actual <- unlist(batch_test[2])
mean_4c_4h_7t<- apply(mc_pred_4con_4hid_7t,1,mean)
y_pred_4c_4h_7t<-ifelse(mean_4c_4h_7t<0.5,0,1)

table(y_pred_4c_4h_7t,y_actual) 
sens_4c_4h_7t <- table(y_pred_4c_4h_7t,y_actual)[1,1]/(table(y_pred_4c_4h_7t,y_actual)[1,1]+
                                                         table(y_pred_4c_4h_7t,y_actual)[2,1])

spec_4c_4h_7t <- table(y_pred_4c_4h_7t,y_actual)[2,2]/(table(y_pred_4c_4h_7t,y_actual)[1,2]+
                                                         table(y_pred_4c_4h_7t,y_actual)[2,2])
# 8t

y_actual <- unlist(batch_test[2])
mean_4c_4h_8t<- apply(mc_pred_4con_4hid_8t,1,mean)
y_pred_4c_4h_8t<-ifelse(mean_4c_4h_8t<0.5,0,1)

table(y_pred_4c_4h_8t,y_actual) 
sens_4c_4h_8t <- table(y_pred_4c_4h_8t,y_actual)[1,1]/(table(y_pred_4c_4h_8t,y_actual)[1,1]+
                                                         table(y_pred_4c_4h_8t,y_actual)[2,1])

spec_4c_4h_8t <- table(y_pred_4c_4h_8t,y_actual)[2,2]/(table(y_pred_4c_4h_8t,y_actual)[1,2]+
                                                         table(y_pred_4c_4h_8t,y_actual)[2,2])

# 9t

y_actual <- unlist(batch_test[2])
mean_4c_4h_9t<- apply(mc_pred_4con_4hid_9t,1,mean)
y_pred_4c_4h_9t<-ifelse(mean_4c_4h_9t<0.5,0,1)

table(y_pred_4c_4h_9t,y_actual) 
sens_4c_4h_9t <- table(y_pred_4c_4h_9t,y_actual)[1,1]/(table(y_pred_4c_4h_9t,y_actual)[1,1]+
                                                         table(y_pred_4c_4h_9t,y_actual)[2,1])

spec_4c_4h_9t <- table(y_pred_4c_4h_9t,y_actual)[2,2]/(table(y_pred_4c_4h_9t,y_actual)[1,2]+
                                                         table(y_pred_4c_4h_9t,y_actual)[2,2])

# 10t

y_actual <- unlist(batch_test[2])
mean_4c_4h_10t<- apply(mc_pred_4con_4hid_10t,1,mean)
y_pred_4c_4h_10t<-ifelse(mean_4c_4h_10t<0.5,0,1)

table(y_pred_4c_4h_10t,y_actual) 
sens_4c_4h_10t <- table(y_pred_4c_4h_10t,y_actual)[1,1]/(table(y_pred_4c_4h_10t,y_actual)[1,1]+
                                                           table(y_pred_4c_4h_10t,y_actual)[2,1])

spec_4c_4h_10t <- table(y_pred_4c_4h_10t,y_actual)[2,2]/(table(y_pred_4c_4h_10t,y_actual)[1,2]+
                                                           table(y_pred_4c_4h_10t,y_actual)[2,2])



sens_spec_4c_4h_all <- cbind(rbind(sens_4c_4h_1t,sens_4c_4h_2n,sens_4c_4h_3r,
                                   sens_4c_4h_4t,sens_4c_4h_5t,sens_4c_4h_6t,
                                   sens_4c_4h_7t,sens_4c_4h_8t,sens_4c_4h_9t,
                                   sens_4c_4h_10t),
                             rbind(spec_4c_4h_1t,spec_4c_4h_2n,spec_4c_4h_3r,
                                   spec_4c_4h_4t,spec_4c_4h_5t,spec_4c_4h_6t,
                                   spec_4c_4h_7t,spec_4c_4h_8t,spec_4c_4h_9t,
                                   spec_4c_4h_10t))

colnames(sens_spec_4c_4h_all) <- c("sens","specs")
rownames(sens_spec_4c_4h_all) <- 1:10

sens_spec_4c_4h_all
apply(sens_spec_4c_4h_all,2,mean)
apply(sens_spec_4c_4h_all,2,sd)




#################
#5con 1hid
################

#1st
y_actual <- unlist(batch_test[2])
mean_5c_1h_1t<- apply(mc_pred_5con_1hid_1t,1,mean)
y_pred_5c_1h_1t<-ifelse(mean_5c_1h_1t<0.5,0,1)

table(y_pred_5c_1h_1t,y_actual) 
sens_5c_1h_1t <- table(y_pred_5c_1h_1t,y_actual)[1,1]/(table(y_pred_5c_1h_1t,y_actual)[1,1]+
                                                         table(y_pred_5c_1h_1t,y_actual)[2,1])

spec_5c_1h_1t <- table(y_pred_5c_1h_1t,y_actual)[2,2]/(table(y_pred_5c_1h_1t,y_actual)[1,2]+
                                                         table(y_pred_5c_1h_1t,y_actual)[2,2])

#2nd

y_actual <- unlist(batch_test[2])
mean_5c_1h_2n<- apply(mc_pred_5con_1hid_2n,1,mean)
y_pred_5c_1h_2n<-ifelse(mean_5c_1h_2n<0.5,0,1)

table(y_pred_5c_1h_2n,y_actual) 
sens_5c_1h_2n <- table(y_pred_5c_1h_2n,y_actual)[1,1]/(table(y_pred_5c_1h_2n,y_actual)[1,1]+
                                                         table(y_pred_5c_1h_2n,y_actual)[2,1])

spec_5c_1h_2n <- table(y_pred_5c_1h_2n,y_actual)[2,2]/(table(y_pred_5c_1h_2n,y_actual)[1,2]+
                                                         table(y_pred_5c_1h_2n,y_actual)[2,2])

# 3r

y_actual <- unlist(batch_test[2])
mean_5c_1h_3r<- apply(mc_pred_5con_1hid_3r,1,mean)
y_pred_5c_1h_3r<-ifelse(mean_5c_1h_3r<0.5,0,1)

table(y_pred_5c_1h_3r,y_actual) 
sens_5c_1h_3r <- table(y_pred_5c_1h_3r,y_actual)[1,1]/(table(y_pred_5c_1h_3r,y_actual)[1,1]+
                                                         table(y_pred_5c_1h_3r,y_actual)[2,1])

spec_5c_1h_3r <- table(y_pred_5c_1h_3r,y_actual)[2,2]/(table(y_pred_5c_1h_3r,y_actual)[1,2]+
                                                         table(y_pred_5c_1h_3r,y_actual)[2,2])

# 4t

y_actual <- unlist(batch_test[2])
mean_5c_1h_4t<- apply(mc_pred_5con_1hid_4t,1,mean)
y_pred_5c_1h_4t<-ifelse(mean_5c_1h_4t<0.5,0,1)

table(y_pred_5c_1h_4t,y_actual) 
sens_5c_1h_4t <- table(y_pred_5c_1h_4t,y_actual)[1,1]/(table(y_pred_5c_1h_4t,y_actual)[1,1]+
                                                         table(y_pred_5c_1h_4t,y_actual)[2,1])

spec_5c_1h_4t <- table(y_pred_5c_1h_4t,y_actual)[2,2]/(table(y_pred_5c_1h_4t,y_actual)[1,2]+
                                                         table(y_pred_5c_1h_4t,y_actual)[2,2])

# 5t

y_actual <- unlist(batch_test[2])
mean_5c_1h_5t<- apply(mc_pred_5con_1hid_5t,1,mean)
y_pred_5c_1h_5t<-ifelse(mean_5c_1h_5t<0.5,0,1)

table(y_pred_5c_1h_5t,y_actual) 
sens_5c_1h_5t <- table(y_pred_5c_1h_5t,y_actual)[1,1]/(table(y_pred_5c_1h_5t,y_actual)[1,1]+
                                                         table(y_pred_5c_1h_5t,y_actual)[2,1])

spec_5c_1h_5t <- table(y_pred_5c_1h_5t,y_actual)[2,2]/(table(y_pred_5c_1h_5t,y_actual)[1,2]+
                                                         table(y_pred_5c_1h_5t,y_actual)[2,2])

# 6t

y_actual <- unlist(batch_test[2])
mean_5c_1h_6t<- apply(mc_pred_5con_1hid_6t,1,mean)
y_pred_5c_1h_6t<-ifelse(mean_5c_1h_6t<0.5,0,1)

table(y_pred_5c_1h_6t,y_actual) 
sens_5c_1h_6t <- table(y_pred_5c_1h_6t,y_actual)[1,1]/(table(y_pred_5c_1h_6t,y_actual)[1,1]+
                                                         table(y_pred_5c_1h_6t,y_actual)[2,1])

spec_5c_1h_6t <- table(y_pred_5c_1h_6t,y_actual)[2,2]/(table(y_pred_5c_1h_6t,y_actual)[1,2]+
                                                         table(y_pred_5c_1h_6t,y_actual)[2,2])

# 7t

y_actual <- unlist(batch_test[2])
mean_5c_1h_7t<- apply(mc_pred_5con_1hid_7t,1,mean)
y_pred_5c_1h_7t<-ifelse(mean_5c_1h_7t<0.5,0,1)

table(y_pred_5c_1h_7t,y_actual) 
sens_5c_1h_7t <- table(y_pred_5c_1h_7t,y_actual)[1,1]/(table(y_pred_5c_1h_7t,y_actual)[1,1]+
                                                         table(y_pred_5c_1h_7t,y_actual)[2,1])

spec_5c_1h_7t <- table(y_pred_5c_1h_7t,y_actual)[2,2]/(table(y_pred_5c_1h_7t,y_actual)[1,2]+
                                                         table(y_pred_5c_1h_7t,y_actual)[2,2])
# 8t

y_actual <- unlist(batch_test[2])
mean_5c_1h_8t<- apply(mc_pred_5con_1hid_8t,1,mean)
y_pred_5c_1h_8t<-ifelse(mean_5c_1h_8t<0.5,0,1)

table(y_pred_5c_1h_8t,y_actual) 
sens_5c_1h_8t <- table(y_pred_5c_1h_8t,y_actual)[1,1]/(table(y_pred_5c_1h_8t,y_actual)[1,1]+
                                                         table(y_pred_5c_1h_8t,y_actual)[2,1])

spec_5c_1h_8t <- table(y_pred_5c_1h_8t,y_actual)[2,2]/(table(y_pred_5c_1h_8t,y_actual)[1,2]+
                                                         table(y_pred_5c_1h_8t,y_actual)[2,2])

# 9t

y_actual <- unlist(batch_test[2])
mean_5c_1h_9t<- apply(mc_pred_5con_1hid_9t,1,mean)
y_pred_5c_1h_9t<-ifelse(mean_5c_1h_9t<0.5,0,1)

table(y_pred_5c_1h_9t,y_actual) 
sens_5c_1h_9t <- table(y_pred_5c_1h_9t,y_actual)[1,1]/(table(y_pred_5c_1h_9t,y_actual)[1,1]+
                                                         table(y_pred_5c_1h_9t,y_actual)[2,1])

spec_5c_1h_9t <- table(y_pred_5c_1h_9t,y_actual)[2,2]/(table(y_pred_5c_1h_9t,y_actual)[1,2]+
                                                         table(y_pred_5c_1h_9t,y_actual)[2,2])

# 10t

y_actual <- unlist(batch_test[2])
mean_5c_1h_10t<- apply(mc_pred_5con_1hid_10t,1,mean)
y_pred_5c_1h_10t<-ifelse(mean_5c_1h_10t<0.5,0,1)

table(y_pred_5c_1h_10t,y_actual) 
sens_5c_1h_10t <- table(y_pred_5c_1h_10t,y_actual)[1,1]/(table(y_pred_5c_1h_10t,y_actual)[1,1]+
                                                           table(y_pred_5c_1h_10t,y_actual)[2,1])

spec_5c_1h_10t <- table(y_pred_5c_1h_10t,y_actual)[2,2]/(table(y_pred_5c_1h_10t,y_actual)[1,2]+
                                                           table(y_pred_5c_1h_10t,y_actual)[2,2])



sens_spec_5c_1h_all <- cbind(rbind(sens_5c_1h_1t,sens_5c_1h_2n,sens_5c_1h_3r,
                                   sens_5c_1h_4t,sens_5c_1h_5t,sens_5c_1h_6t,
                                   sens_5c_1h_7t,sens_5c_1h_8t,sens_5c_1h_9t,
                                   sens_5c_1h_10t),
                             rbind(spec_5c_1h_1t,spec_5c_1h_2n,spec_5c_1h_3r,
                                   spec_5c_1h_4t,spec_5c_1h_5t,spec_5c_1h_6t,
                                   spec_5c_1h_7t,spec_5c_1h_8t,spec_5c_1h_9t,
                                   spec_5c_1h_10t))

colnames(sens_spec_5c_1h_all) <- c("sens","specs")
rownames(sens_spec_5c_1h_all) <- 1:10

sens_spec_5c_1h_all
apply(sens_spec_5c_1h_all,2,mean)
apply(sens_spec_5c_1h_all,2,sd)





#################
#5con 2hid
################

#1st
y_actual <- unlist(batch_test[2])
mean_5c_2h_1t<- apply(mc_pred_5con_2hid_1t,1,mean)
y_pred_5c_2h_1t<-ifelse(mean_5c_2h_1t<0.5,0,1)

table(y_pred_5c_2h_1t,y_actual) 
sens_5c_2h_1t <- table(y_pred_5c_2h_1t,y_actual)[1,1]/(table(y_pred_5c_2h_1t,y_actual)[1,1]+
                                                         table(y_pred_5c_2h_1t,y_actual)[2,1])

spec_5c_2h_1t <- table(y_pred_5c_2h_1t,y_actual)[2,2]/(table(y_pred_5c_2h_1t,y_actual)[1,2]+
                                                         table(y_pred_5c_2h_1t,y_actual)[2,2])

#2nd

y_actual <- unlist(batch_test[2])
mean_5c_2h_2n<- apply(mc_pred_5con_2hid_2n,1,mean)
y_pred_5c_2h_2n<-ifelse(mean_5c_2h_2n<0.5,0,1)

table(y_pred_5c_2h_2n,y_actual) 
sens_5c_2h_2n <- table(y_pred_5c_2h_2n,y_actual)[1,1]/(table(y_pred_5c_2h_2n,y_actual)[1,1]+
                                                         table(y_pred_5c_2h_2n,y_actual)[2,1])

spec_5c_2h_2n <- table(y_pred_5c_2h_2n,y_actual)[2,2]/(table(y_pred_5c_2h_2n,y_actual)[1,2]+
                                                         table(y_pred_5c_2h_2n,y_actual)[2,2])

# 3r

y_actual <- unlist(batch_test[2])
mean_5c_2h_3r<- apply(mc_pred_5con_2hid_3r,1,mean)
y_pred_5c_2h_3r<-ifelse(mean_5c_2h_3r<0.5,0,1)

table(y_pred_5c_2h_3r,y_actual) 
sens_5c_2h_3r <- table(y_pred_5c_2h_3r,y_actual)[1,1]/(table(y_pred_5c_2h_3r,y_actual)[1,1]+
                                                         table(y_pred_5c_2h_3r,y_actual)[2,1])

spec_5c_2h_3r <- table(y_pred_5c_2h_3r,y_actual)[2,2]/(table(y_pred_5c_2h_3r,y_actual)[1,2]+
                                                         table(y_pred_5c_2h_3r,y_actual)[2,2])

# 4t

y_actual <- unlist(batch_test[2])
mean_5c_2h_4t<- apply(mc_pred_5con_2hid_4t,1,mean)
y_pred_5c_2h_4t<-ifelse(mean_5c_2h_4t<0.5,0,1)

table(y_pred_5c_2h_4t,y_actual) 
sens_5c_2h_4t <- table(y_pred_5c_2h_4t,y_actual)[1,1]/(table(y_pred_5c_2h_4t,y_actual)[1,1]+
                                                         table(y_pred_5c_2h_4t,y_actual)[2,1])

spec_5c_2h_4t <- table(y_pred_5c_2h_4t,y_actual)[2,2]/(table(y_pred_5c_2h_4t,y_actual)[1,2]+
                                                         table(y_pred_5c_2h_4t,y_actual)[2,2])

# 5t

y_actual <- unlist(batch_test[2])
mean_5c_2h_5t<- apply(mc_pred_5con_2hid_5t,1,mean)
y_pred_5c_2h_5t<-ifelse(mean_5c_2h_5t<0.5,0,1)

table(y_pred_5c_2h_5t,y_actual) 
sens_5c_2h_5t <- table(y_pred_5c_2h_5t,y_actual)[1,1]/(table(y_pred_5c_2h_5t,y_actual)[1,1]+
                                                         table(y_pred_5c_2h_5t,y_actual)[2,1])

spec_5c_2h_5t <- table(y_pred_5c_2h_5t,y_actual)[2,2]/(table(y_pred_5c_2h_5t,y_actual)[1,2]+
                                                         table(y_pred_5c_2h_5t,y_actual)[2,2])

# 6t

y_actual <- unlist(batch_test[2])
mean_5c_2h_6t<- apply(mc_pred_5con_2hid_6t,1,mean)
y_pred_5c_2h_6t<-ifelse(mean_5c_2h_6t<0.5,0,1)

table(y_pred_5c_2h_6t,y_actual) 
sens_5c_2h_6t <- table(y_pred_5c_2h_6t,y_actual)[1,1]/(table(y_pred_5c_2h_6t,y_actual)[1,1]+
                                                         table(y_pred_5c_2h_6t,y_actual)[2,1])

spec_5c_2h_6t <- table(y_pred_5c_2h_6t,y_actual)[2,2]/(table(y_pred_5c_2h_6t,y_actual)[1,2]+
                                                         table(y_pred_5c_2h_6t,y_actual)[2,2])

# 7t

y_actual <- unlist(batch_test[2])
mean_5c_2h_7t<- apply(mc_pred_5con_2hid_7t,1,mean)
y_pred_5c_2h_7t<-ifelse(mean_5c_2h_7t<0.5,0,1)

table(y_pred_5c_2h_7t,y_actual) 
sens_5c_2h_7t <- table(y_pred_5c_2h_7t,y_actual)[1,1]/(table(y_pred_5c_2h_7t,y_actual)[1,1]+
                                                         table(y_pred_5c_2h_7t,y_actual)[2,1])

spec_5c_2h_7t <- table(y_pred_5c_2h_7t,y_actual)[2,2]/(table(y_pred_5c_2h_7t,y_actual)[1,2]+
                                                         table(y_pred_5c_2h_7t,y_actual)[2,2])
# 8t

y_actual <- unlist(batch_test[2])
mean_5c_2h_8t<- apply(mc_pred_5con_2hid_8t,1,mean)
y_pred_5c_2h_8t<-ifelse(mean_5c_2h_8t<0.5,0,1)

table(y_pred_5c_2h_8t,y_actual) 
sens_5c_2h_8t <- table(y_pred_5c_2h_8t,y_actual)[1,1]/(table(y_pred_5c_2h_8t,y_actual)[1,1]+
                                                         table(y_pred_5c_2h_8t,y_actual)[2,1])

spec_5c_2h_8t <- table(y_pred_5c_2h_8t,y_actual)[2,2]/(table(y_pred_5c_2h_8t,y_actual)[1,2]+
                                                         table(y_pred_5c_2h_8t,y_actual)[2,2])

# 9t

y_actual <- unlist(batch_test[2])
mean_5c_2h_9t<- apply(mc_pred_5con_2hid_9t,1,mean)
y_pred_5c_2h_9t<-ifelse(mean_5c_2h_9t<0.5,0,1)

table(y_pred_5c_2h_9t,y_actual) 
sens_5c_2h_9t <- table(y_pred_5c_2h_9t,y_actual)[1,1]/(table(y_pred_5c_2h_9t,y_actual)[1,1]+
                                                         table(y_pred_5c_2h_9t,y_actual)[2,1])

spec_5c_2h_9t <- table(y_pred_5c_2h_9t,y_actual)[2,2]/(table(y_pred_5c_2h_9t,y_actual)[1,2]+
                                                         table(y_pred_5c_2h_9t,y_actual)[2,2])

# 10t

y_actual <- unlist(batch_test[2])
mean_5c_2h_10t<- apply(mc_pred_5con_2hid_10t,1,mean)
y_pred_5c_2h_10t<-ifelse(mean_5c_2h_10t<0.5,0,1)

table(y_pred_5c_2h_10t,y_actual) 
sens_5c_2h_10t <- table(y_pred_5c_2h_10t,y_actual)[1,1]/(table(y_pred_5c_2h_10t,y_actual)[1,1]+
                                                           table(y_pred_5c_2h_10t,y_actual)[2,1])

spec_5c_2h_10t <- table(y_pred_5c_2h_10t,y_actual)[2,2]/(table(y_pred_5c_2h_10t,y_actual)[1,2]+
                                                           table(y_pred_5c_2h_10t,y_actual)[2,2])



sens_spec_5c_2h_all <- cbind(rbind(sens_5c_2h_1t,sens_5c_2h_2n,sens_5c_2h_3r,
                                   sens_5c_2h_4t,sens_5c_2h_5t,sens_5c_2h_6t,
                                   sens_5c_2h_7t,sens_5c_2h_8t,sens_5c_2h_9t,
                                   sens_5c_2h_10t),
                             rbind(spec_5c_2h_1t,spec_5c_2h_2n,spec_5c_2h_3r,
                                   spec_5c_2h_4t,spec_5c_2h_5t,spec_5c_2h_6t,
                                   spec_5c_2h_7t,spec_5c_2h_8t,spec_5c_2h_9t,
                                   spec_5c_2h_10t))

colnames(sens_spec_5c_2h_all) <- c("sens","specs")
rownames(sens_spec_5c_2h_all) <- 1:10

sens_spec_5c_2h_all
apply(sens_spec_5c_2h_all,2,mean)
apply(sens_spec_5c_2h_all,2,sd)

################################################
################################################


# saving all history
history_df_do_3c_5h_1t <- as.data.frame(history_do_3c_5h_1t)
history_df_do_3c_5h_2n <- as.data.frame(history_do_3c_5h_2n)
history_df_do_3c_5h_3r <- as.data.frame(history_do_3c_5h_3r)
history_df_do_3c_5h_4t <- as.data.frame(history_do_3c_5h_4t)
history_df_do_3c_5h_5t <- as.data.frame(history_do_3c_5h_5t)
history_df_do_3c_5h_6t <- as.data.frame(history_do_3c_5h_6t)
history_df_do_3c_5h_7t <- as.data.frame(history_do_3c_5h_7t)
history_df_do_3c_5h_8t <- as.data.frame(history_do_3c_5h_8t)
history_df_do_3c_5h_9t <- as.data.frame(history_do_3c_5h_9t)
history_df_do_3c_5h_10t <- as.data.frame(history_do_3c_5h_10t)


history_df_do_3c_4h_1t <- as.data.frame(history_do_3c_4h_1t)
history_df_do_3c_4h_2n <- as.data.frame(history_do_3c_4h_2n)
history_df_do_3c_4h_3r <- as.data.frame(history_do_3c_4h_3r)
history_df_do_3c_4h_4t <- as.data.frame(history_do_3c_4h_4t)
history_df_do_3c_4h_5t <- as.data.frame(history_do_3c_4h_5t)
history_df_do_3c_4h_6t <- as.data.frame(history_do_3c_4h_6t)
history_df_do_3c_4h_7t <- as.data.frame(history_do_3c_4h_7t)
history_df_do_3c_4h_8t <- as.data.frame(history_do_3c_4h_8t)
history_df_do_3c_4h_9t <- as.data.frame(history_do_3c_4h_9t)
history_df_do_3c_4h_10t <- as.data.frame(history_do_3c_4h_10t)


history_df_do_4c_5h_1t <- as.data.frame(history_do_4c_5h_1t)
history_df_do_4c_5h_2n <- as.data.frame(history_do_4c_5h_2n)
history_df_do_4c_5h_3r <- as.data.frame(history_do_4c_5h_3r)
history_df_do_4c_5h_4t <- as.data.frame(history_do_4c_5h_4t)
history_df_do_4c_5h_5t <- as.data.frame(history_do_4c_5h_5t)
history_df_do_4c_5h_6t <- as.data.frame(history_do_4c_5h_6t)
history_df_do_4c_5h_7t <- as.data.frame(history_do_4c_5h_7t)
history_df_do_4c_5h_8t <- as.data.frame(history_do_4c_5h_8t)
history_df_do_4c_5h_9t <- as.data.frame(history_do_4c_5h_9t)
history_df_do_4c_5h_10t <- as.data.frame(history_do_4c_5h_10t)


history_df_do_4c_4h_1t <- as.data.frame(history_do_4c_4h_1t)
history_df_do_4c_4h_2n <- as.data.frame(history_do_4c_4h_2n)
history_df_do_4c_4h_3r <- as.data.frame(history_do_4c_4h_3r)
history_df_do_4c_4h_4t <- as.data.frame(history_do_4c_4h_4t)
history_df_do_4c_4h_5t <- as.data.frame(history_do_4c_4h_5t)
history_df_do_4c_4h_6t <- as.data.frame(history_do_4c_4h_6t)
history_df_do_4c_4h_7t <- as.data.frame(history_do_4c_4h_7t)
history_df_do_4c_4h_8t <- as.data.frame(history_do_4c_4h_8t)
history_df_do_4c_4h_9t <- as.data.frame(history_do_4c_4h_9t)
history_df_do_4c_4h_10t <- as.data.frame(history_do_4c_4h_10t)


history_df_do_5c_1h_1t <- as.data.frame(history_do_5c_1h_1t)
history_df_do_5c_1h_2n <- as.data.frame(history_do_5c_1h_2n)
history_df_do_5c_1h_3r <- as.data.frame(history_do_5c_1h_3r)
history_df_do_5c_1h_4t <- as.data.frame(history_do_5c_1h_4t)
history_df_do_5c_1h_5t <- as.data.frame(history_do_5c_1h_5t)
history_df_do_5c_1h_6t <- as.data.frame(history_do_5c_1h_6t)
history_df_do_5c_1h_7t <- as.data.frame(history_do_5c_1h_7t)
history_df_do_5c_1h_8t <- as.data.frame(history_do_5c_1h_8t)
history_df_do_5c_1h_9t <- as.data.frame(history_do_5c_1h_9t)
history_df_do_5c_1h_10t <- as.data.frame(history_do_5c_1h_10t)


history_df_do_5c_2h_1t <- as.data.frame(history_do_5c_2h_1t)
history_df_do_5c_2h_2n <- as.data.frame(history_do_5c_2h_2n)
history_df_do_5c_2h_3r <- as.data.frame(history_do_5c_2h_3r)
history_df_do_5c_2h_4t <- as.data.frame(history_do_5c_2h_4t)
history_df_do_5c_2h_5t <- as.data.frame(history_do_5c_2h_5t)
history_df_do_5c_2h_6t <- as.data.frame(history_do_5c_2h_6t)
history_df_do_5c_2h_7t <- as.data.frame(history_do_5c_2h_7t)
history_df_do_5c_2h_8t <- as.data.frame(history_do_5c_2h_8t)
history_df_do_5c_2h_9t <- as.data.frame(history_do_5c_2h_9t)
history_df_do_5c_2h_10t <- as.data.frame(history_do_5c_2h_10t)



#saving all plot history
plot_3c_5h_1t <- plot(history_do_3c_5h_1t)
plot_3c_5h_2n <-plot(history_do_3c_5h_2n)
plot_3c_5h_3r <-plot(history_do_3c_5h_3r)
plot_3c_5h_4t <-plot(history_do_3c_5h_4t)
plot_3c_5h_5t <-plot(history_do_3c_5h_5t)
plot_3c_5h_6t <-plot(history_do_3c_5h_6t)
plot_3c_5h_7t <-plot(history_do_3c_5h_7t)
plot_3c_5h_8t <-plot(history_do_3c_5h_8t)
plot_3c_5h_9t <-plot(history_do_3c_5h_9t)
plot_3c_5h_10t <-plot(history_do_3c_5h_10t)


plot_3c_4h_1t <- plot(history_do_3c_4h_1t)
plot_3c_4h_2n <-plot(history_do_3c_4h_2n)
plot_3c_4h_3r <-plot(history_do_3c_4h_3r)
plot_3c_4h_4t <-plot(history_do_3c_4h_4t)
plot_3c_4h_5t <-plot(history_do_3c_4h_5t)
plot_3c_4h_6t <-plot(history_do_3c_4h_6t)
plot_3c_4h_7t <-plot(history_do_3c_4h_7t)
plot_3c_4h_8t <-plot(history_do_3c_4h_8t)
plot_3c_4h_9t <-plot(history_do_3c_4h_9t)
plot_3c_4h_10t <-plot(history_do_3c_4h_10t)



plot_4c_5h_1t <- plot(history_do_4c_5h_1t)
plot_4c_5h_2n <-plot(history_do_4c_5h_2n)
plot_4c_5h_3r <-plot(history_do_4c_5h_3r)
plot_4c_5h_4t <-plot(history_do_4c_5h_4t)
plot_4c_5h_5t <-plot(history_do_4c_5h_5t)
plot_4c_5h_6t <-plot(history_do_4c_5h_6t)
plot_4c_5h_7t <-plot(history_do_4c_5h_7t)
plot_4c_5h_8t <-plot(history_do_4c_5h_8t)
plot_4c_5h_9t <-plot(history_do_4c_5h_9t)
plot_4c_5h_10t <-plot(history_do_4c_5h_10t)


plot_4c_4h_1t <- plot(history_do_4c_4h_1t)
plot_4c_4h_2n <-plot(history_do_4c_4h_2n)
plot_4c_4h_3r <-plot(history_do_4c_4h_3r)
plot_4c_4h_4t <-plot(history_do_4c_4h_4t)
plot_4c_4h_5t <-plot(history_do_4c_4h_5t)
plot_4c_4h_6t <-plot(history_do_4c_4h_6t)
plot_4c_4h_7t <-plot(history_do_4c_4h_7t)
plot_4c_4h_8t <-plot(history_do_4c_4h_8t)
plot_4c_4h_9t <-plot(history_do_4c_4h_9t)
plot_4c_4h_10t <-plot(history_do_4c_4h_10t)


plot_5c_1h_1t <- plot(history_do_5c_1h_1t)
plot_5c_1h_2n <-plot(history_do_5c_1h_2n)
plot_5c_1h_3r <-plot(history_do_5c_1h_3r)
plot_5c_1h_4t <-plot(history_do_5c_1h_4t)
plot_5c_1h_5t <-plot(history_do_5c_1h_5t)
plot_5c_1h_6t <-plot(history_do_5c_1h_6t)
plot_5c_1h_7t <-plot(history_do_5c_1h_7t)
plot_5c_1h_8t <-plot(history_do_5c_1h_8t)
plot_5c_1h_9t <-plot(history_do_5c_1h_9t)
plot_5c_1h_10t <-plot(history_do_5c_1h_10t)



plot_5c_2h_1t <- plot(history_do_5c_2h_1t)
plot_5c_2h_2n <-plot(history_do_5c_2h_2n)
plot_5c_2h_3r <-plot(history_do_5c_2h_3r)
plot_5c_2h_4t <-plot(history_do_5c_2h_4t)
plot_5c_2h_5t <-plot(history_do_5c_2h_5t)
plot_5c_2h_6t <-plot(history_do_5c_2h_6t)
plot_5c_2h_7t <-plot(history_do_5c_2h_7t)
plot_5c_2h_8t <-plot(history_do_5c_2h_8t)
plot_5c_2h_9t <-plot(history_do_5c_2h_9t)
plot_5c_2h_10t <-plot(history_do_5c_2h_10t)





