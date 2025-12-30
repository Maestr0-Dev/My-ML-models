import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm #support vector machine
from sklearn.metrics import accuracy_score
from config import data_path

diabetes_data=pd.read_csv(data_path+'diabetes-1.csv')
# print(diabetes_data.describe())
# print(diabetes_data['Outcome'].value_counts())

# print(diabetes_data.groupby('Outcome').mean())

#/////////////Preparing the data//////////////////////
data=diabetes_data.drop(columns='Outcome',axis=1)
labels=diabetes_data['Outcome']
#Data standardization
scaler=StandardScaler()
scaler.fit(data)
standardized_data=scaler.transform(data)

new_data=standardized_data

data_train, data_test, labels_train, labels_test=train_test_split(new_data,labels,test_size=0.15,stratify=labels,random_state=2)


#/////////////Training the model//////////////////////
start_time=time.perf_counter()
classifier=svm.SVC(kernel='linear')
# Training the support vector machine classifier
classifier.fit(data_train,labels_train)
end_time=time.perf_counter()
training_duration=end_time-start_time
# Model evaluation
data_train_prediction=classifier.predict(data_train)
training_data_accuracy=accuracy_score(data_train_prediction,labels_train)
# print('Accuracy score on training data : ', training_data_accuracy)
# print('Training duration is:',training_duration)
 
data_test_prediction=classifier.predict(data_test)
test_data_accuracy=accuracy_score(data_test_prediction,labels_test)
# print('Accuracy score on test data : ', test_data_accuracy)

# Predictive system
input_data=(5,166,72,19,175,25.8,0.587,51)
np_input_data=np.asarray(input_data)
input_data_reshape=np_input_data.reshape(1,-1) 
# Standardizing the data as done during the training phase
std_data=scaler.transform(input_data_reshape)
prediction=classifier.predict(std_data)
if (prediction[0]==0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')