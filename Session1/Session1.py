#Remember --> Syntax may vary from one version to another

import tensorflow as tf
print(tf.__version__)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(D,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#What the fuck am I doing here?
# We have two layers:
# 1) What we call the input object --> We do it in order to tell Keras the size of our input vector 'x'
# 2) Then we have to create the Dense object --> Keras knows the size becuase it was specified in the input, we just need to specify the output size (1 in this case) and the activation function!

#See we put this two specifications into a list and pass it to the keras model "Sequential"
#Sequential model means that each layer in the list will happen sequentaly, meaning, one after the other

#Train/fit our model
model.compile(optimizer='adam',           # The optimizer for Deep Learning is by default 'adam' (most of the times)
              loss='binary_crossentropy', # See more on this later --> Always used for binary classification
              metrics=['accuracy']
)

#If all our predictions are exactly correct --> Cost function will be cero
# The more incorrect our prediction --> The larger the cost function will become

# !ls --> All files in current directory