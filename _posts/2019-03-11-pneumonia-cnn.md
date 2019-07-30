---
layout:     post
title:      "Using Convolutional Neural Networks to Predict Pneumonia"
date:       2019-03-11 12:00:00
author:     "A.I. Dan"
---

# Using Convolutional Neural Networks to Predict Pneumonia

This blog post will start with a brief introduction and overview of convolutional neural networks and will then transition over to applying this new knowledge by predicting pneumonia from x-ray images with an accuracy of over 92%. While such an accuracy is nothing to get too excited about, it is a respectable result for such a simple convolutional neural network.

When it comes time to show the code examples, the code will be shown first, then below each code example there will be an explanation about the code.

<b>Dataset</b>:

[Pneumonia X-Ray Dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2)

## A Brief History of CNNs and the Visual Cortex

Convolutional Neural Networks (CNNs), or ConvNets, are neural networks that are commonly used for image and audio recognition and classification. CNNs stem from an animal brain's visual cortex. Studies have shown that monkey's and cat's visual cortexes have neurons that respond to small subfields of the visual field. Every neuron is responsible for a small section of the visual field, called the receptive field. Together, all neurons in the visual cortex will cover the entire visual space [(Hubel, 1968)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1557912/pdf/jphysiol01104-0228.pdf).

The human brain's visual cortex is made up of columns of neurons that share a similar function. An array of these neuronal columns make up what is called a module [(Neuroscientifically Challenged, 2016)](https://www.neuroscientificallychallenged.com/blog/know-your-brain-primary-visual-cortex). Each module is capable of responding to only a small subsection of the visual field and therefore, the visual cortex consists of many of these modules to cover the entire area. While this is not exactly how our convolutional neural network will function, there will be noticeable similarities to an animal's visual cortex.

## A Brief Introduction to CNNs

Like all common neural networks, CNNs have neurons with adjustable weights and biases. Normal neural networks are fully connected, meaning that every neuron is connected to every neuron from the previous layer. CNNs are not fully connected like normal neural networks though, as his would be too computationally expensive and is simply not needed to achieve the desired results. Using a fully connected neural network would not be very efficient when dealing with image data with large input sizes.

To imagine the large number of parameters, think of our chest X-ray images. These images will have an input shape of 64x64x3, or 64 wide, 64 high, with 3 color channels. If a fully connected neural network were to be used, this would mean that a single neuron in a single hidden layer would consist of 12,288 connections (64 x 64 x 3 = 12,288) [(CS231n, 2018)](http://cs231n.github.io/convolutional-networks/). This is with only one fully connected neuron. Imagine the number of all the weights in a neural network with many neurons! It is easy to understand why fully connected neural networks would not be the most efficient method of classifying images. This is where CNNs come in handy, except a CNN's architecture does in fact include a few fully connected layer(s).

## A Brief Introduction to a CNN's Architecture
![arch](a-i-dan.github.io/images/cnn_arch.png?raw=true)

Like all neural networks, CNNs have an input and output layer with a number of hidden layers that will apply an activation function, typically ReLu. A CNNs design will consist of three main layers: Convolutional layer, pooling layer, and the fully connected layer. Each layer will be covered below:

### Convolutional Layer:

The convolutional layer is responsible for finding and extractng features from the input data. Convolutional layers use filters, also called kernels, for this feature extracting process. Since CNNs are not fully connected, neurons are only connected to a predetermined subregion of the input space. The size of this region is called the filter size, or receptive field. The receptive field of a neuron is simply the space that it will receive inputs from.

For this example, we will be using a filter size of 3x3. We only set the width and height of the receptive field because the depth of the filter must be the same depth as the input and is automatically set. In our case, our input has 3 color channels. Therefore, the input's depth is 3. This means that each neuron in this convolutional layer will have 27 weights (3x3x3 = 27).

A convolutional layer convolves the input by sliding these filters around the input space while computing the dot product of the weights and inputs. The pixels within the filter will be converted to a single value that will represent the entire receptive field.

<img src='a-i-dan.github.io/images/output_TMXYGX.gif?raw=true' style='margin: auto; width: 500px; display: block;'>

### Pooling Layer:

Pooling layers, otherwise known as downsampling layers, will mostly be seen following convolutional layers of the neural network. The job of the pooling layer is to reduce the spatial dimensions of the input. This will result in a reduced number of parameters and will also help our model generalize and avoid overfitting. This blog post will be using <b>max pooling</b>, the most commonly used type of pooling layer. There are other versions of the pooling layer such as average pooling, but the focus of this post will be on max pooling.

<b>Max Pooling:</b> The convolutional layer will find a specific feature in a region within the input and will assign it a higher activation value. The pooling layer will then reduce this region and create a new representation. The max pooling layer essentially creates an abstraction of the original region by using the max values found in each subregion.

Max pooling will sweep over each subregion, apply a max filter that will extract the highest value from each subregion and create an abstraction with reduced dimensions.

The example below shows a 4x4 matrix as our input. We will be using a 2x2 filter to sweep over our input matrix and we will also be using a stride of 2. The 2x2 pool size, or filter, will determine the amount by which we downscale the spatial dimensions. For a 2x2 pool size, we will downscale by half each time. The stride will determine the amount of steps to move while scanning the input matrix. For example, with a stride of 2, we will scan the input matrix from the red 2x2 to the green 2x2, etc. The region being scanned will move two blocks over each time.  

<!--![](a-i-dan.github.io/images/CNN_Figures.png?raw=true?raw=true)
![](a-i-dan.github.io/images/CNN_Figures-2.png?raw=true)-->
<img src='a-i-dan.github.io/images/CNN_Figures.png?raw=true' style='margin:auto; width: 500px; display: block;'>

<img src='a-i-dan.github.io/images/CNN_Figures-2.png?raw=true' style='margin:auto; width: 500px; display: block;'>



### Fully Connected Layer:

Like normal neural networks, each neuron in the fully connected layer of a CNN is connected to every neuron in the previous layer. The fully connected layers are responsible for classifying the data after feature extraction. The fully connected layer will look at the activation maps of high-level features created by the convolutional layers or pooling layers and will then determine which features are associated with each class.

For our dataset, we have two classes: pneumonia and normal. The fully connected layers will look at the features the previous layers have found and will then determine which features best help predict the class the image will fall under.

## A Brief Introduction to Pneumonia

Every year in the United States alone, about one million people will visit the hospital due to pneumonia. Of those one million people, about 50,000 people will die from pneumonia each year [(CDC, 2017)](https://www.cdc.gov/pneumonia/prevention.html?CDC_AA_refVal=https%3A%2F%2Fwww.cdc.gov%2Ffeatures%2Fpneumonia%2Findex.html).

Pneumonia is an infectious inflammatory disease that affects the lungs of people of all ages and is typically caused by viral or bacterial infections. Pneumonia affects one or both sides of of the lungs and causes the alveoli (air sacs) to fill up with fluid, bacteria, microorganisms and pus [(NIH, 2018)]((https://www.nhlbi.nih.gov/health-topics/pneumonia)).  

Pneumonia is diagnosed in many ways, one common way of confirmation is through chest X-rays. Chest X-rays are the best tests, and most accurate, to determine if one has pneumonia. While it is crucial, detecting pneumonia can sometimes be a difficult task. Pneumonia often vaguely shows up in X-rays and can also get mixed in with other diseases present in that local area.


<!--![](a-i-dan.github.io/images/iu.jpg?raw=true)-->
<img src='a-i-dan.github.io/images/iu.jpg?raw=true' style='margin: auto; width: 400px; display: block;'>

<center><a href='https://proxy.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.mznEecI8B-fpfXm8fEQymAHaGN%26pid%3D15.1&f=1'>Image Source</a></center>

## Data Preparation and Analysis

The first portion of the code will be dedicated to preparing the data. This section will be less about the details than the actual building of the model. I put all of the code in this <i>Data Preparation and Analysis</i> section, excluding the visuals, in a separate file called <a href='pneumonia_dataset.py'>pneumonia_dataset</a> for later importing in the <i>Applying CNNs to Predicting Pneumonia</i> section. Importing the dataset from this file will be explained at the beginning of the section.


```python
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from keras.preprocessing.image import ImageDataGenerator
```

The first few lines are importing the libraries we will need for preparing and visualizing our data.


```python
path  = "./chest_xray"
dirs  = os.listdir(path)
print(dirs)
```
    Output:

    ['.DS_Store', 'test', 'train', 'val']


Here we are setting the path of the chest_xray folder for later use. We are then printing out the directories from within the chest_xray folder. Notice that the folder is split into three subfolders: test, train and val, or validation. Each folder contains chest X-ray images that we will need to use for training and testing.


```python
train_folder = path + '/train/'
test_folder  = path + '/test/'
val_folder   = path + '/val/'

train_dirs = os.listdir(train_folder)
print(train_dirs)
```
    Output:

    ['.DS_Store', 'PNEUMONIA', 'NORMAL']


Next, we will set the paths for each folder. We can use the "path" variable we set earlier and concatenate that with each subfolder's name. We will then want to see the contents of the training folder. To view the directories, we will use the `listdir()` function for the training folder, then print the results.


```python
train_normal = train_folder + 'NORMAL/'
train_pneu   = train_folder + 'PNEUMONIA/'
```

We can then take our training folder and and set the paths to each class. In this case, we have two classes: the <b>normal</b> images and the <b>pneumonia</b> images. If we want to visualize images that are specifically "normal" or "pneumonia", then we will create a variable that contains the path to these images for later reference.


```python
pneu_images   = glob(train_pneu + "*.jpeg")
normal_images = glob(train_normal + "*.jpeg")
```

Now that we have split the training folder into "normal" and "pneumonia", we can pull all of the images out of each class. The images in this dataset are all jpeg images, so for each path we will add `.jpeg` at the end to make sure we are pulling out the images.


```python
def show_imgs(num_of_imgs):

    for img in range(num_of_imgs):
        pneu_pic   = np.asarray(plt.imread(pneu_images[img]))
        normal_pic = np.asarray(plt.imread(normal_images[img]))

        fig = plt.figure(figsize= (15,10))

        normal_plot = fig.add_subplot(1,2,1)
        plt.imshow(normal_pic, cmap='gray')
        normal_plot.set_title('Normal')
        plt.axis('off')

        pneu_plot = fig.add_subplot(1, 2, 2)
        plt.imshow(pneu_pic, cmap='gray')
        pneu_plot.set_title('Pneumonia')
        plt.axis('off')

        plt.show()
```

We will create a function called `show_imgs()` to visualize the chest X-ray images from within our training set. The function will take one argument that specifies how many images to show (`num_of_imgs`). We will then use a for loop with a range of "num_of_imgs" to show however many images is specified.

We will be showing <b>normal</b> images and <b>pneumonia</b> images side by side so we will add two sub plots: one for normal, one for pneumonia. The color map for these images will be 'grays'. If you feel like changing the color map, head over to Matplotlb's [color map reference](https://matplotlib.org/examples/color/colormaps_reference.html) page.

For each image shown, we will label it as either "normal" or "pneumonia" by setting each sub plot's title.


```python
show_imgs(3)
```

![png](a-i-dan.github.io/images/pneumonia_blog_post_17_0.png?raw=true)



![png](a-i-dan.github.io/images/pneumonia_blog_post_17_1.png?raw=true)



![png](a-i-dan.github.io/images/pneumonia_blog_post_17_2.png?raw=true)


We can use our `show_imgs()` function like this. We will call the function and give it one argument: the number of images of both classes we would like to show.  


```python
train_datagen = ImageDataGenerator(rescale            = 1/255,
                                   shear_range        = 0.2,
                                   zoom_range         = 0.2,
                                   horizontal_flip    = True,
                                   rotation_range     = 40,
                                   width_shift_range  = 0.2,
                                   height_shift_range = 0.2)
```

This is called <b>image preprocessing</b>, or <b>data augmentation</b>. We will be using the `ImageDataGenerator()` class from Keras for our data augmentation. Data augmentation helps us to expand our training dataset. The more training data and the more variety the better. With more training data and with slightly manipulated data, overfitting becomes less of a problem as our model has to generalize more.

* The first step is to rescale our data. Rescaling images is a common practice because most images have RGB values ranging from 0-255. These values are too high for most models to handle, but by multiplying these values by 1/255, we can condense each RGB value to a value between 0-1. This is much easier for our model to process.

* Next we have `shear_range` which will randomly apply shear mapping, or shear transformations to the data. The value "0.2" is the [shear](https://en.wikipedia.org/wiki/Shear_mapping) intensity, or shear angle.

* `zoom_range` is also set to "0.2". This is for randomly zooming in on the images.

* `horizontal_flip` is set to "True" because we want to randomly flip half of the images in our dataset.

* `rotation_range` is the value in degrees for which the image may be randomly rotated.

* `width_shift_range` and `height_shift_range` are ranges for randomly translating images.


```python
test_datagen = ImageDataGenerator(rescale = 1/255)
```

This is where we rescale our test set. The test set does not need all of the same transformations applied to the training data. Only the training data can be manipulated to avoid overfitting. The test set must be the original images to accurately predict pneumonia on real, minimally manipulated, images.


```python
training_set = train_datagen.flow_from_directory(train_folder,
                                   target_size= (64, 64),
                                   batch_size = 32,
                                   class_mode = 'binary')

val_set = test_datagen.flow_from_directory(val_folder,
                                   target_size=(64, 64),
                                   batch_size = 32,
                                   class_mode ='binary')

test_set = test_datagen.flow_from_directory(test_folder,
                                   target_size= (64, 64),
                                   batch_size = 32,
                                   class_mode = 'binary')
```
    Output:

    Found 5216 images belonging to 2 classes.
    Found 16 images belonging to 2 classes.
    Found 624 images belonging to 2 classes.


Now we will take the path of our test, train, and validation folders and generate batches of augmented data using `flow_from_directory()` from Keras.
* The first argument will be the directory to pull from.

* The second argument is the target size, or the dimensions of the images after they are resized.

* The third argument is "class_mode", which is set to "binary". This will return 1D binary labels. This dataset calls for binary classification due to the fact that there are only two classes.

Now that we are done preparing our data, we can move on to building the model, training it, then testing it and getting our results in the form of accuracy scores.

## Applying CNNs to Predicting Pneumonia


```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout

import pneumonia_dataset
```

[Keras](https://keras.io) is a high-level python neural network library that runs on top of TensorFlow. Keras enables quick and efficient implementation and experimentation of deep learning and machine learning algorithms while still being very effective. Keras will be our deep learning library of choice for this blog post so we will be importing a few required layers and models to make our convolutional neural network function well.

The last import statement is the <a href='pneumonia_dataset.py'>pneumonia_dataset</a> file that I mentioned earlier in the <i>Data Preparation and Analysis</i> section.


```python
training_set, test_set, val_set = pneumonia_dataset.load_data()
```
    Output:

    Training Set:
    Found 5216 images belonging to 2 classes.

    Validation Set:
    Found 16 images belonging to 2 classes.

    Test Set:
    Found 624 images belonging to 2 classes.


The pneumonia_dataset file will return a training set, test set and a validation set that we will appropriately name. This will return a summary of how our data is split up amongst each set, including the number of images in each set and how many classes these images will fit into.


```python
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))
```

This is the exciting part.

* First, we create our model using the "Sequential" model from Keras. This model is a linear stack of layers, meaning that we will create our model layer-by-layer.

* **1st Convolutional Layer:**
The first convolutional layer is our input layer.

    * The first parameter is the amount of convolutional filters to use in the layer, which is set to "32". This is also the number of neurons, or nodes, that will be in this layer.

    * The second parameter is the filter's size, or the receptive field. Imagine we are creating a window of the size (3, 3), or a width of three and a height of three, that our convolutional layer is restricted to looking through at any given time.

    * The third parameter we will set is the activation function. Our nonlinear activation function is ReLu, or rectified linear unit. The ReLu function is <i>`f(x) = max(0, x)`</i>. Therefore, all negatives are converted to zeros while all positives remain the same. ReLu is one of the most popular activation functions because it reduces the the vanishing gradient issue and is computationally cheaper to compute. This does not mean that the ReLu function is perfect, but it will get the job done for most applications.

    * The fourth parameter is the input shape. This parameter <b>only needs to be specified in the first convolutional layer</b>. After the first layer, our model can handle the rest. The input shape is simply the shape of the images that will be fed to the CNN. The shape of our input images will be (64, 64, 3) (width, height, depth).

    * The final parameter is the padding, which is set to "same". This will pad the input in a way that makes the output have the same length as the initial input.  

* **1st Max Pooling Layer:**
The max pooling layers will only have one parameter for this model.

    * The parameter is the pool size, or the factor to downscale the input's spatial dimensions. The pool size will be set to (2, 2), which will downscale by half each time. Refer to the earlier section "A Brief Introduction to a CNN's Architecture" for more details about the pooling layer.

* **2nd Convolutional and Max Pooling Layer:**
The second convolutional layer and max pooling layer will be the same as the previous layers above. The second convolutional layer will not need the input size to be specified.

* **3rd Convolutional Layer:**
In the third convolutional layer, the first parameter will be changed. In the first two convolutional layers, the number of filters, or neurons in the layer, was set to "32", but for the third layer it will be set to "64". Other than this one change, everything else will stay the same.

* **3rd Max Pooling Layer:**
The third max pooling layer will be the same as the first two previous pooling layers.

* **Flatten:**
Flattening is required to convert multi-dimensional data into usable data for the fully connected layers. In order for the fully connected layers to work, we need to convert the convolutional layer's output to a 1D vector. Our convolutional layers will be using 2D data (images). This will have to be reshaped, or flattened, to one dimension before it is fed into the classifier.

    * If we take a look at a portion of the model summary, the output data of the third max pooling layer has a shape of `(None, 6, 6, 64)`. The output shape after flattening is `(None, 2304)`. This is because (6 * 6 * 64) = 2304.

    ```
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    max_pooling2d_16 (MaxPooling (None, 6, 6, 64)          0         
    _________________________________________________________________
    flatten_5 (Flatten)          (None, 2304)              0         
    _________________________________________________________________
    ```

* **Dense - ReLu:**
 Dense layers are the fully connected layers, meaning that every neuron is connected to all the neurons in previous layers. We will be using 128 nodes. This also means that the fully connected layer with have an output size of 128. For this fully connected layer, the ReLu activation function will be used.

* **Dropout:**
 Dropout is used to regularize our model and reduce overfitting. Dropout will temporarily "drop out" random nodes in the fully connected layers. This dropping out of nodes will result in a thinned neural network that consists of the nodes that were not dropped. Dropout reduces overfitting and helps the model generalize due to the fact that no specific node can be 100% reliable. The ".5" means that the probability of a certain node being dropped is 50%. To read more about dropout, check out [this paper](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf).

* **Dense - Sigmoid:**
 Our final fully connected layer will use the <b>sigmoid function</b>. Our problem involves two classes: Pneumonia and normal. This is a binary classification problem where sigmoid can be used to return a probability between 0 and 1. If this were a multi-class classification, the sigmoid activation function would not be the weapon of choice. However, for this simple model, the sigmoid function works just fine. The sigmoid function can be defined as:

<img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;f(x)=\frac{1}{1&space;&plus;&space;e^{-x}}" title="f(x)=\frac{1}{1 + e^{-x}}" style='margin: auto; display: block;'>

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

We can now configure the model using the compile method from Keras.

* The first argument is the optimizer which will be set to "adam". The adam optimizer is one of the most popular algorithms in deep learning right now. The authors of [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980v8) state that Adam combines the advantages of two other popular optimizers: RMSProp and AdaGrad. You can read about the effectiveness of Adam for CNNs in section 6.3 of the Adam paper.

* The second argument is the loss function. This model will use the <b>binary cross entropy</b> loss function. Our model will be conducting binary classification, so we can write this loss function as shown below, where "y" is either 0 or 1, indicating if the class label is the correct classification and where "p" is the model's predicted probability:

<img src="https://latex.codecogs.com/gif.latex?-(y\log(p)&space;&plus;&space;(1&space;-&space;y)\log(1&space;-&space;p))" title="-(y\log(p) + (1 - y)\log(1 - p))" style='margin: auto; display: block;'>


* The last argument is the metric function that will judge the performance of the model. In this case, we want the accuracy to be returned.


```python
model_train = model.fit_generator(training_set,
                         steps_per_epoch = 200,
                         epochs = 5,
                         validation_data = val_set,
                         validation_steps = 100)
```
    Output:

    Epoch 1/5
    200/200 [==============================] - 139s 697ms/step - loss: 0.2614 - acc: 0.8877 - val_loss: 0.5523 - val_acc: 0.8125
    Epoch 2/5
    200/200 [==============================] - 124s 618ms/step - loss: 0.2703 - acc: 0.8811 - val_loss: 0.5808 - val_acc: 0.8125
    Epoch 3/5
    200/200 [==============================] - 124s 618ms/step - loss: 0.2448 - acc: 0.8984 - val_loss: 0.7902 - val_acc: 0.8125
    Epoch 4/5
    200/200 [==============================] - 121s 607ms/step - loss: 0.2444 - acc: 0.8955 - val_loss: 0.8172 - val_acc: 0.7500
    Epoch 5/5
    200/200 [==============================] - 119s 597ms/step - loss: 0.2177 - acc: 0.9092 - val_loss: 0.8556 - val_acc: 0.6250


It is now time to train the model! This will be done using the `fit_generator()` method from Keras. This will train the model on batches of data that are generated from the training set.

* The first argument is the steps per epoch. This will be set to 200. The steps per epoch will tell the model the total number of batches of samples to produce from the generator before concluding that specific epoch.

* The second argument is the number of epochs, or training iterations. The Keras  documentation states that an epoch is defined as an iteration over the entire data provided, as defined by steps_per_epoch.

* The third argument is the validation data the model will use. The model will not be trained on the validation data, but this will help measure the loss at the end of every epoch.

* The final argument is the validation steps. our validation data is coming from a generator (see above code), so the number of batches of samples to produce from the generator must be set, similar to the steps per epoch.

```python
test_accuracy = model.evaluate_generator(test_set,steps=624)

print('Testing Accuracy: {:.2f}%'.format(test_accuracy[1] * 100))
```
    Output:

    Testing Accuracy: 90.22%


Now that the model has been trained, it is time to evaluate the model's accuracy on the test data. This will be done by using the `evaluate_generator()` method from Keras. This evaluation will return the test set loss and accuracy results.

* Just like the fit generator, the first argument for the evaluate generator is the folder from which to pull samples from. Since we are testing our model's accuracy, the test set will be used.

* The second argument is the number of batches of samples to pull from the generator before finishing.

We can then print the accuracy and shorten it to only show two decimal places. The accuracy will be returned as a value between 0-1, so we will multiply it by 100 to receive the percentage.

After that, the model is complete! We have had some success with predicting pneumonia from chest X-ray images!


## Everything Put Together (20 Epochs)


```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Dense, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator

path  = "./chest_xray"

train_folder = path + '/train/'
test_folder  = path + '/test/'
val_folder   = path + '/val/'

train_datagen = ImageDataGenerator(rescale            = 1/255,
                                   shear_range        = 0.2,
                                   zoom_range         = 0.2,
                                   horizontal_flip    = True,
                                   rotation_range     = 40,
                                   width_shift_range  = 0.2,
                                   height_shift_range = 0.2)

test_datagen = ImageDataGenerator(rescale = 1/255)

training_set = train_datagen.flow_from_directory(train_folder,              
                                   target_size = (64, 64),
                                   batch_size  = 32,
                                   class_mode  = 'binary')

val_generator = test_datagen.flow_from_directory(val_folder,              
                                   target_size =(64, 64),
                                   batch_size  =32,
                                   class_mode  ='binary')

test_set = test_datagen.flow_from_directory(test_folder,         
                                   target_size = (64, 64),
                                   batch_size  = 32,
                                   class_mode  = 'binary')


model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_train = model.fit_generator(training_set,
                         steps_per_epoch = 200,
                         epochs = 20,
                         validation_data = val_set,
                         validation_steps = 100)

test_accuracy = model.evaluate_generator(test_set,steps=624)

print('Testing Accuracy: {:.2f}%'.format(test_accuracy[1] * 100))
```
```
Output:

Found 5216 images belonging to 2 classes.
Found 16 images belonging to 2 classes.
Found 624 images belonging to 2 classes.
Epoch 1/20
200/200 [==============================] - 142s 708ms/step - loss: 0.5141 - acc: 0.7369 - val_loss: 0.6429 - val_acc: 0.6250
Epoch 2/20
200/200 [==============================] - 137s 683ms/step - loss: 0.4034 - acc: 0.8058 - val_loss: 0.6182 - val_acc: 0.7500
Epoch 3/20
200/200 [==============================] - 134s 670ms/step - loss: 0.3334 - acc: 0.8483 - val_loss: 0.6855 - val_acc: 0.6875
Epoch 4/20
200/200 [==============================] - 129s 644ms/step - loss: 0.3337 - acc: 0.8516 - val_loss: 0.8377 - val_acc: 0.6875
Epoch 5/20
200/200 [==============================] - 139s 696ms/step - loss: 0.3012 - acc: 0.8672 - val_loss: 0.6252 - val_acc: 0.8750
Epoch 6/20
200/200 [==============================] - 132s 662ms/step - loss: 0.2719 - acc: 0.8808 - val_loss: 0.6599 - val_acc: 0.6875
Epoch 7/20
200/200 [==============================] - 125s 627ms/step - loss: 0.2503 - acc: 0.8969 - val_loss: 0.6470 - val_acc: 0.7500
Epoch 8/20
200/200 [==============================] - 128s 638ms/step - loss: 0.2347 - acc: 0.9016 - val_loss: 0.8703 - val_acc: 0.6875
Epoch 9/20
200/200 [==============================] - 131s 656ms/step - loss: 0.2337 - acc: 0.9075 - val_loss: 0.6313 - val_acc: 0.6875
Epoch 10/20
200/200 [==============================] - 124s 619ms/step - loss: 0.2159 - acc: 0.9133 - val_loss: 0.7781 - val_acc: 0.7500
Epoch 11/20
200/200 [==============================] - 129s 647ms/step - loss: 0.1962 - acc: 0.9228 - val_loss: 0.6118 - val_acc: 0.8125
Epoch 12/20
200/200 [==============================] - 127s 634ms/step - loss: 0.1826 - acc: 0.9306 - val_loss: 0.5831 - val_acc: 0.8125
Epoch 13/20
200/200 [==============================] - 128s 638ms/step - loss: 0.2071 - acc: 0.9178 - val_loss: 0.4661 - val_acc: 0.8125
Epoch 14/20
200/200 [==============================] - 124s 619ms/step - loss: 0.1902 - acc: 0.9234 - val_loss: 0.6944 - val_acc: 0.7500
Epoch 15/20
200/200 [==============================] - 128s 638ms/step - loss: 0.1763 - acc: 0.9281 - val_loss: 0.6350 - val_acc: 0.6875
Epoch 16/20
200/200 [==============================] - 139s 696ms/step - loss: 0.1727 - acc: 0.9337 - val_loss: 0.4813 - val_acc: 0.8750
Epoch 17/20
200/200 [==============================] - 145s 724ms/step - loss: 0.1689 - acc: 0.9334 - val_loss: 0.3188 - val_acc: 0.7500
Epoch 18/20
200/200 [==============================] - 133s 664ms/step - loss: 0.1650 - acc: 0.9366 - val_loss: 0.4164 - val_acc: 0.8750
Epoch 19/20
200/200 [==============================] - 132s 661ms/step - loss: 0.1755 - acc: 0.9316 - val_loss: 0.5974 - val_acc: 0.8125
Epoch 20/20
200/200 [==============================] - 132s 662ms/step - loss: 0.1616 - acc: 0.9395 - val_loss: 0.4295 - val_acc: 0.8750
Testing Accuracy: 92.13%
```

## Experimenting with Input Depth/Color Channels

While writing this post I became curious if the input's color channels made much of a difference in overall accuracy. I decided to add this litte side experiment to the post because I think it is interesting and may influence future blog posts. Everything in the code below is the same except for a few minor changes. For the training set, test set and validation set generator, a parameter had to be added to convert the images to grayscale.

For each dataset generator we will have to add `color_mode = 'grayscale'` along with the other parameters. Then the final change occurs in the first convolutional layer when the input shape is set. Originally, with three color channels, the input shape was set to (64, 64, 3), but now that we are converting the images to grayscale, there will only be one color channel. The new input shape will be set to be <b>(64, 64, 1)</b>. After those changes have been made, the model is ready to run!


```python
train_datagen = ImageDataGenerator(rescale            = 1/255,
                                   shear_range        = 0.2,
                                   zoom_range         = 0.2,
                                   horizontal_flip    = True,
                                   rotation_range     = 40,
                                   width_shift_range  = 0.2,
                                   height_shift_range = 0.2)

test_datagen = ImageDataGenerator(rescale = 1/255)

training_set = train_datagen.flow_from_directory('./chest_xray/train/',
                                   color_mode = "grayscale",
                                   target_size= (64, 64),
                                   batch_size = 32,
                                   class_mode = 'binary')

val_set = test_datagen.flow_from_directory('./chest_xray/val/',
                                   color_mode = "grayscale",
                                   target_size=(64, 64),
                                   batch_size = 32,
                                   class_mode ='binary')

test_set = test_datagen.flow_from_directory('./chest_xray/test/',
                                   color_mode = "grayscale",
                                   target_size= (64, 64),
                                   batch_size = 32,
                                   class_mode = 'binary')
```


```python
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_train = model.fit_generator(training_set,
                         steps_per_epoch = 200,
                         epochs = 5,
                         validation_data = val_set,
                         validation_steps = 100)

test_accuracy = model.evaluate_generator(test_set,steps=624)

print('Testing Accuracy: {:.2f}%'.format(test_accuracy[1] * 100))
```

```
Output:

Epoch 1/5
200/200 [==============================] - 125s 626ms/step - loss: 0.5203 - acc: 0.7481 - val_loss: 0.5989 - val_acc: 0.7500
Epoch 2/5
200/200 [==============================] - 108s 539ms/step - loss: 0.3987 - acc: 0.8039 - val_loss: 0.5855 - val_acc: 0.8125
Epoch 3/5
200/200 [==============================] - 100s 500ms/step - loss: 0.3333 - acc: 0.8408 - val_loss: 0.5360 - val_acc: 0.6875
Epoch 4/5
200/200 [==============================] - 98s 489ms/step - loss: 0.3084 - acc: 0.8591 - val_loss: 0.7180 - val_acc: 0.7500
Epoch 5/5
200/200 [==============================] - 111s 554ms/step - loss: 0.2848 - acc: 0.8762 - val_loss: 0.6304 - val_acc: 0.7500
Testing Accuracy: 89.26%
```

It seems like the color channels of the input images does not matter as much as I originally thought. I ran both input shapes (64, 64, 3) and (64, 64, 1) and the results seemed to stay around the same general area. This dataset is most likely not the best dataset to use for RBG vs grayscale experiments. After running this for 20 epochs, the grayscale model achieved a 91.67% test set accuracy which is just below what the original model achieved.


## Conclusion

The convolutional neural network acheived a 92.13% accuracy on the test set. You can determine for yourself if that should be labeled as a "success". For a simple model, I would consider it to be pretty reasonable.

I initially thought that reducing the input's color channels would result in a higher accuracy due to the cheaper computational costs and an overall simpler model. I was proven wrong in this example. I plan on reading into this and playing around with some different examples. I may write a follow up post about comparing RGB images and grayscale images in convolutional neural networks.
