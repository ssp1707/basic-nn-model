# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The Neural network model contains input layer, two hidden layers and output layer. Input layer contains a single neuron. Output layer also contains single neuron. First hidden layer contains six neurons and second hidden layer contains four neurons. A neuron in input layer is connected with every neurons in a first hidden layer. Similarly, each neurons in first hidden layer is connected with all neurons in second hidden layer. All neurons in second hidden layer is connected with output layered neuron. Relu activation function is used here and the model is a linear neural network model. Data is the key for the working of neural network and we need to process it before feeding to the neural network. In the first step, we will visualize data which will help us to gain insight into the data. We need a neural network model. This means we need to specify the number of hidden layers in the neural network and their size, the input and output size. Now we need to define the loss function according to our task. We also need to specify the optimizer to use with learning rate. Fitting is the training step of the neural network. Here we need to define the number of epochs for which we need to train the neural network. After fitting model, we can test it on test data to check whether the case of overfitting.

## Neural Network Model

![image](https://user-images.githubusercontent.com/75235167/187084520-1af19950-cff6-4683-81ba-5b2665968baa.png)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```
Developed By:S.Sanjna Priya
Registration Number: 212220230043
```

```python3

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
import matplotlib.pyplot as plt

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('DL Data').sheet1

rows = worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'Input':'float','Output':'float'})

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df.head()

x=df[['Input']].values
x

y=df[['Output']].values
y

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=11)

Scaler=MinMaxScaler()
Scaler.fit(x_train)
Scaler.fit(x_test)
x_train1=Scaler.transform(x_train)
x_test1=Scaler.transform(x_test)
x_train1

ai_brain = Sequential([
    Dense(6,activation='relu'),
    Dense(4,activation='relu'),
    Dense(1)
])

ai_brain.compile(
    optimizer='rmsprop',
    loss='mse'
)
ai_brain.fit(x_train1,y_train,epochs=4000)

loss_df=pd.DataFrame(ai_brain.history.history)
loss_df.plot()
plt.title('Training Loss Vs Iteration Plot')

ai_brain.evaluate(x_test1,y_test)

x_n1=[[66]]
x_n1_1=Scaler.transform(x_n1)
ai_brain.predict(x_n1_1)
```
## Dataset Information

![E1 0](https://user-images.githubusercontent.com/75234965/187119910-0c2b3461-cba9-47fc-9a70-f7930176fe79.PNG)

## OUTPUT

### Training Loss Vs Iteration Plot

![E1 1](https://user-images.githubusercontent.com/75234965/187120262-1531b3a8-d780-483d-9163-8ef0a0c27a37.PNG)

### Test Data Root Mean Squared Error

![image](https://user-images.githubusercontent.com/75234965/187120317-8cba881a-8b24-499e-8eed-f6d36ca7b522.png)

### New Sample Data Prediction

![E1 3](https://user-images.githubusercontent.com/75234965/187120386-36370204-9cac-4de4-a554-bf6be2034ffc.PNG)

## RESULT

Succesfully created and trained a neural network regression model for the given dataset.
