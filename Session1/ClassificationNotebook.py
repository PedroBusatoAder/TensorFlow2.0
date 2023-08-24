# We will be developing a classification model to detect whether the tumor is malignant o benign
import tensorflow as tf
from sklearn.datasets import load_breast_cancer  #Scikit-learn es una biblioteca de Python que proporciona acceso a algoritmos y datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


#Save the data in our variable!
data = load_breast_cancer()     # It is a Bunch object --> Similar to a dictionary
#print(data.keys())              # We can access al the keys in our dictionary/bunch!
#print(data.data.shape)          # We can see the amount of samples by number of features
#print(data.feature_names)       # Our features/columns

#print(data.target)              # We said we use this dataset for binary classification --> We see how we classify the tumors in 0 and 1
#print(data.target_names)        # We verify that 0 means 'malignant' and 1 'benign'

# Now let us begin!
# 1) Train and test data split
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)
N, D = x_train.shape             # We save on 'N' the amount of samples and on 'D' the amount of columns/attributes
print(N,D)

# 2) Preprocess the data by scaling it --> Standarizes all samples by substracting their mean and then divide over its standard deviation
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) # fit_transform fits and transforms in a single step
x_test = scaler.transform(x_test)       # We now just transform data since we want to emulate and evaluate how the model works in the real world

# 3) Move on to our TensorFlow steps
# i) Create our lineal model --> Classification model based on Keras
model = tf.keras.models.Sequential([                    # We learn why we use the Keras API later on the course
    tf.keras.layers.Input(shape=(D,)),
    tf.keras.layers.Dense(1, activation='sigmoid'),      # 1  since we have just one output
])

# ii) We compile our model  --> Check explanation on Google Docs
model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
)

# iii) Now we need to train our model
r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100) # We save our training on a variable since it returns something!
print('Train score:', model.evaluate(x_train, y_train))
print('Test score:', model.evaluate(x_test, y_test))

# iv) We plot the loss and accuracy per epoch
plt.plot(r.history['loss'], label='loss')               # We plot training loss
plt.plot(r.history['val_loss'], label='val_loss')       # We plot testing loss

plt.plot(r.history['accuracy'], label='acc')            # We plot training accuracy
plt.plot(r.history['val_accuracy'], label='val_acc')    # We plot testing accuracy

plt.legend()
plt.show()

print('Done!')

