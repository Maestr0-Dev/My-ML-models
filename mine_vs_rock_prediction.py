import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading the dataset to pd dataframe
path='C:/Users/Mene/Desktop/All/Learning Coding/Python/ML/mine_prediction/'
sonar_data=pd.read_csv(path+'sonar_data.csv',header=None) #this means there is no header in the csv file
# print(sonar_data.head())
# print(sonar_data.describe())
# print(sonar_data[60].value_counts()) #to see how many of each label we have on 60th column

sonar_data.groupby(60).mean() #to see the mean of each column based on the labels on the 60th column

#/////////////Preparing the data//////////////////////
X=sonar_data.drop(columns=60,axis=1) #dropping the 60th column(1) for row its 0
Y=sonar_data[60]
#splitting  into training and test data

X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)
#stratify=Y is used to make sure that both training and test data have similar(alomost equal) distribution of labels Y(M and R)
# print(X.shape, X_train.shape, X_test.shape)

#uisng logistic regression tp train the model

model=LogisticRegression()
model.fit(X_train,Y_train) #training the logistic regression model with training data

#accuracy on training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print('Accuracy score on training data : ', training_data_accuracy)

#accuracy on test data
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print('Accuracy score on test data : ', test_data_accuracy)

#predicting whether the object is a rock or a mine based on the input data given by the user
input_data=(0.1083,0.1070,0.0257,0.0837,0.0748,0.1125,0.3322,0.4590,0.5526,0.5966,0.5304,0.2251,0.2402,0.2689,0.6646,0.6632,0.1674,0.0837,0.4331,0.8718,0.7992,0.3712,0.1703,0.1611,0.2086,0.2847,0.2211,0.6134,0.5807,0.6925,0.3825,0.4303,0.7791,0.8703,1.0000,0.9212,0.9386,0.9303,0.7314,0.4791,0.2087,0.2016,0.1669,0.2872,0.4374,0.3097,0.1578,0.0553,0.0334,0.0209,0.0172,0.0180,0.0110,0.0234,0.0276,0.0032,0.0084,0.0122,0.0082,0.0143)

np_input_data=np.asarray(input_data)
input_data_reshape=np_input_data.reshape(1,-1) #reshaping the data as we are predicting for one instance

prediction=model.predict(input_data_reshape)
# print(prediction)
# if (prediction[0]=='R'):
#     print('The object is a Rock')
# else:
#     print('The object is a Mine')