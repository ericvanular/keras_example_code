"""keras_example_code.py

Demonstrative code showing how to compile and use a Sequential Model in Keras
"""

#Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as keras
from sklearn.model_selection import train_test_split

#Create a random normally distributed feature matrix with 4 feature columns & 100 rows
X1 = np.random.randn(100, 4)

##y1 = Should equal 1 if the value of corresponding mean(X1[i]) >= 0 otherwise, should equal to 0
y1=np.zeros(100)
for idx,row in enumerate(X1):
  if(np.mean(X1[idx]) >= 0):
    y1[idx]=1
  else:
    y1[idx]=0

#Repeat for another 100x4 matrix
X2 = np.random.randn(100, 4)

## y2 = Should equal 1 if the value of corresponding mean(X2[i]) >= 0 otherwise, should equal to 0
y2=np.zeros(100)
for idx,row in enumerate(X2):
  if(np.mean(X2[idx]) >= 0):
    y2[idx]=1
  else:
    y2[idx]=0

# Concatenate the two arrays X1, and X2, Resulting shape should be (200, 4)
X = np.concatenate((X1, X2), axis=0) 

# Concatenate the two arrays y1, and y2, Resulting shape should be (200)
y = np.concatenate((y1, y2), axis=0) 

#Making sure y has the right shape after concatnation
print(y.shape)

"""## Explore the data (Using dataframes)

- Look for stats within the data 
- Plot the features
- Plot a histogram of the feature 1 (index: 0) with 10 buckets
- Plot a histogram of the y values ( 2 buckets )
- Any other data insights
"""

df = pd.DataFrame(X)
df.head()

plt.hist(y,bins=2)
plt.show()

plt.hist(X,bins=10)
plt.show()

df.describe()

"""## Build a model

Using Keras let's build a logistic regressor
"""

y = onehotencode(y)
def model():
  # create model
  model = keras.Sequential()
  # Add layer
  model.add(keras.layers.Dense(32, input_shape(,4)))
  model.add(keras.layers.Dense(2, activation='softmax'))
  # Compile model
  # Choose a loss function and an optimizer
  model.compile(loss='mean_squared_error', optimizer='adam')
  return model

"""## Splitting the data into train, test"""

X_train, X_test, y_train, y_test = train_test_split(X, y test_size=0.2, random_state=10)

"""## Train the evaluate the model"""

num_epochs = 20
batch_size = 10

model().fit(x, y, batch_size=batch_size, epochs=num_epochs) 

model().evaluate(x, y, batch_size=batch_size, epochs=num_epochs)