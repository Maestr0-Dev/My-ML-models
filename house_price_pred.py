import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns
import  sklearn.datasets
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split

# from xgboost import XGBRegressor

house_data=sklearn.datasets.fetch_california_housing()

house_dataframe=pd.DataFrame(house_data.data,columns=house_data.feature_names)# without the column attribute, the columns will not have name, only numbers
house_dataframe['Price']=house_data.target
# print(house_dataframe.isnull().sum())
# print(house_dataframe.describe())

#/////////////Understanding the correlation between various features in the dataset//////////////////////
correlation=house_dataframe.corr()
#creating a heatmap to understand the correlation
plt.figure(figsize=(6,6))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
# plt.show()
data=house_dataframe.drop(columns='Price',axis=1)
labels=house_dataframe['Price']

data_train, data_test, labels_train, labels_test=train_test_split(data,labels,test_size=0.15,random_state=2)

start=time.perf_counter()
#Model training
model= XGBRegressor()
model.fit(data_train,labels_train)
end=time.perf_counter()
time=end-start
print("training duration",time)

#accuracy for prediction
test_data_prediction=model.predict(data_test)
 # print(training_data_prediction)

 
 #R squared error 
score_1=metrics.r2_score(labels_test,test_data_prediction)
#mean absolute error
score_2=metrics.mean_absolute_error(labels_test,test_data_prediction)

print("R squared error:", score_1)
print("Mean Absolute Error :",score_2)