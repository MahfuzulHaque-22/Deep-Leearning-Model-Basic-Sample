# Deep-Leearning-Model-Basic-Sample



we start by defining the input shapes for the image and text data, as well as the number of output classes. We then define the number of hidden layers we want to use.

We create the image input layer using the Input class from TensorFlow, and then create a series of convolutional layers using the Conv2D and MaxPooling2D layers. We then flatten the output of the convolutional layers using the Flatten layer.

We create the text input layer using the Input class, and then create an embedding layer and LSTM layer for the text data.

We concatenate the outputs of the image and text layers using the concatenate function from TensorFlow.

We create the hidden layers using a for loop, adding each layer to a list called hidden_layers. We use the Dense layer from TensorFlow, which creates a fully connected layer of neurons. We set the number of neurons and activation function for each layer.

Finally, we create the output layer, which also uses the Dense layer with the softmax activation function (since we're doing a classification task with 10 possible outputs). We then create the model using the Model class from TensorFlow, with the input and output layers we defined earlier.

We compile the model using the adam optimizer and categorical_crossentropy loss function (since we're doing a multi-class classification task). We also specify that we want to track accuracy as a metric.
