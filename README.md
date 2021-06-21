# files
efficientEmotionDetectionWithKeras is messy w/ a lot of what I experimented with left in

provenModel and model2_efficient_keras are used to store the models that are used in this project

cleanModel.py is a cleaned up version of the model (easier to read, but just the model)

# emotionDetector
emotion detector using data from https://www.kaggle.com/astraszab/facial-expression-dataset-image-folders-fer2013

# data
data can be found from kaggle link above, the data folder is in the same directory as the python file

# description

- we start with an ImageDataGenerator that is used on the training data, adding rotations, zooms, shears, shifts and brightness changes to create a more robust data set
- we use the flow_from_directory so that we can more easily process the large amounts of training data
- we import a bunch of stuff to use for the models
- we create the model
- getAllData() is used to get data from all images in a folder (used for val/test)
- the checkpoint is used to save the model based on validation accurqacy
- get_f1 is an additional metric (metrics = [cost, accuracy, get_f1]) to consider with the model
- if we are just testing the model we compare the validation metrics with the test metrics
- we load an image so we can successfully set the input/outputsize of our model
- if we are supposed to load a saved model, we do so (saving is based on weights)
- we use plt and mpatches to plot metrics for the model, helping us debug as we consider the validation cost/accuracy/f1 versus the training cost/accuracy/f1
- we fit the model and plot the corresponding values (see above) on graph


# model
- the model begins with broad convolutions of increasing size
- we start with a same convolution then continue to valid
- we have very little maxPooling because the images are smaller and it was determined not to have an effect (at least in how I employed it)
- we have several dense layers conencted to the final convolution, including Dropout for regularization as well as some l2 regularization. Note that we also use flat/X in the previous so that the additional model layers do not prevent a hinderance (@ least on the training data)
