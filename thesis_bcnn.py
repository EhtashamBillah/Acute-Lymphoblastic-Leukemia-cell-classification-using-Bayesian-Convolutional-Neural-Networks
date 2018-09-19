
# coding: utf-8

# In[18]:


from keras import Input, layers, Model,optimizers
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
import os, shutil


# In[21]:


# The path to the directory where the original
# dataset was uncompressed
#original_dataset_dir = '/Users/fchollet/Downloads/kaggle_original_data'
"""
# The directory where we will
# store our smaller dataset
base_dir = 'C:/Users/Ehtasham/Dropbox/Ehtasham/leukemia/ALL_IDB2'

# Directories for our training,
# validation and test splits

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# Directory with our training cat pictures
train_yes_dir = os.path.join(train_dir, 'yes')
os.mkdir(train_yes_dir)

# Directory with our training dog pictures
train_no_dir = os.path.join(train_dir, 'no')
os.mkdir(train_no_dir)

# Directory with our validation cat pictures
validation_yes_dir = os.path.join(validation_dir, 'yes')
os.mkdir(validation_yes_dir)

# Directory with our validation dog pictures
validation_no_dir = os.path.join(validation_dir, 'no')
os.mkdir(validation_no_dir)

# Directory with our validation cat pictures
test_yes_dir = os.path.join(test_dir, 'yes')
os.mkdir(test_yes_dir)

# Directory with our validation dog pictures
test_no_dir = os.path.join(test_dir, 'no')
os.mkdir(test_no_dir)
"""

# Copy first 1000 cat images to train_cats_dir
fnames = ['Im{}_1.tif'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

# Copy next 500 cat images to validation_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 500 cat images to test_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)
# Copy first 1000 dog images to train_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 500 dog images to validation_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 500 dog images to test_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)


# In[2]:


do_rate_conv = 0.2
do_rate_hid = 0.5

input_tensor = Input(shape = (257,257,3))
x = layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(input_tensor)
x = layers.Dropout(do_rate_conv)(x, training=True)
x = layers.MaxPool2D(pool_size= (2,2))(x)
x = layers.Conv2D(filters=64 ,kernel_size=(3,3), padding='same', activation='relu')(x)
x = layers.Dropout(do_rate_conv)(x, training=True)
x = layers.MaxPool2D(pool_size= (2,2))(x)
x = layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')(x)
x = layers.Dropout(do_rate_conv)(x, training=True)
x = layers.MaxPool2D(pool_size= (2,2))(x)
x = layers.Flatten()(x)
x = layers.Dense(units=1024, activation='relu')(x)
x = layers.Dropout(do_rate_hid)(x, training=True)
x = layers.Dense(units=1024, activation='relu')(x)
x = layers.Dropout(do_rate_hid)(x, training=True)
x = layers.Dense(units=1024, activation='relu')(x)
x = layers.Dropout(do_rate_hid)(x, training=True)
x = layers.Dense(units=1024, activation='relu')(x)
x = layers.Dropout(do_rate_hid)(x, training=True)
x = layers.Dense(units=1024, activation='relu')(x)
x = layers.Dropout(do_rate_hid)(x, training=True)
output_tensor = layers.Dense(units=1, activation= 'sigmoid')(x)


# In[3]:


model_3c_5h = Model(inputs = input_tensor,outputs = output_tensor)
model_3c_5h.summary()


# In[4]:


import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
plot_model(model_3c_5h, show_shapes=True, to_file='BCNN_3C_5H.png')


# In[14]:


model_3c_5h.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                   loss = "binary_crossentropy",
                   metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255) # tried validation_split = 0.2
test_datagen = ImageDataGenerator(rescale = 1./255) 

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size = (257,257),
                                                    batch_size = 20,
                                                    class_mode = 'binary',                                                    
                                                    shuffle=True)

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                    target_size = (257,257),
                                                    batch_size = 20,
                                                    class_mode = 'binary',                                                    
                                                    shuffle=True)


# In[ ]:



                                                   

