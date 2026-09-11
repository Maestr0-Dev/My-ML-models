import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LogicRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv('./Datasets/RT_IOT2022')
polished_data=data.drop(['Unnamed: 0', 'bwd_URG_flag_count'],axis=1)
X=polished_data.drop(['Attack_type'], axis=1)
y=polished_data['Attack_type']

X = pd.get_dummies(X, columns=["proto", "service"], dtype=int)
# print(X[['proto_icmp', 'proto_tcp', 'proto_udp', 'service_-', 
#         'service_dhcp', 'service_dns', 'service_http', 'service_irc',
#         'service_mqtt', 'service_ntp', 'service_radius', 'service_ssh', 'service_ssl']
#       .isnull().sum().head(30))
trainX,testX,trainY,testY= train_test_split(X,y,test_size=0.2,random_state=42, stratify=y)
# print(trainX.columns.tolist())


  # Logic Regression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# numerical=trainX.drop(['proto_icmp', 'proto_tcp', 'proto_udp', 'service_-', 'service_dhcp', 'service_dns', 'service_http', 'service_irc', 'service_mqtt', 'service_ntp', 'service_radius', 'service_ssh', 'service_ssl'],axis=1)
# scaler.fit(numerical)
# num=scaler.transform(numerical)
# num=pd.DataFrame(num, columns=numerical.columns, index=numerical.index)
# lr_trainX=pd.concat([num, trainX[['proto_icmp', 'proto_tcp', 'proto_udp', 'service_-', 'service_dhcp', 'service_dns', 'service_http', 'service_irc', 'service_mqtt', 'service_ntp', 'service_radius', 'service_ssh', 'service_ssl']]], axis=1)

# numerical=testX.drop(['proto_icmp', 'proto_tcp', 'proto_udp', 'service_-', 'service_dhcp', 'service_dns', 'service_http', 'service_irc', 'service_mqtt', 'service_ntp', 'service_radius', 'service_ssh', 'service_ssl'],axis=1)
# num=scaler.transform(numerical)
# num=pd.DataFrame(num, columns=numerical.columns, index=numerical.index)
# lr_testX=pd.concat([num, testX[['proto_icmp', 'proto_tcp', 'proto_udp', 'service_-', 'service_dhcp', 'service_dns', 'service_http', 'service_irc', 'service_mqtt', 'service_ntp', 'service_radius', 'service_ssh', 'service_ssl']]], axis=1)


# model=LogicRegressor()
# model.fit(lr_trainX,trainY)

# predictions=model.predict(lr_testX)

# for i in range(1,30):
#     print(f"Predicted: {predictions[i]}, Actual: {testY.iloc[i]}")

# Accuracy_score=[accuracy_score(testY, predictions)]
# Classification_report=[classification_report(testY, predictions)]
# F1=[f1_score(testY, predictions, average='macro')]
# Confusion_matrix=[confusion_matrix(testY, predictions)] 
# print("Logistic Regression Results:")
# print(f"Accuracy: {Accuracy_score[0]}")
# print(f"F1 Score: {F1[0]}")
# print("Classification Report:")
# print(Classification_report[0])
# print("Confusion Matrix:")
# print(Confusion_matrix[0])


# Decision tree
tree_model=DecisionTreeClassifier()
tree_model.fit(trainX,trainY)
tree_predictions=tree_model.predict(testX)

Accuracy_score=accuracy_score(testY, tree_predictions)
Classification_report=classification_report(testY, tree_predictions)
F1=f1_score(testY, tree_predictions, average='macro')
Confusion_matrix=confusion_matrix(testY, tree_predictions)

# print("Decision Tree Results:")
# print(f"Accuracy: {Accuracy_score}")
# print(f"F1 Score: {F1}")
# print("Classification Report:")
# print(Classification_report)
# print("Confusion Matrix:")
# print(Confusion_matrix)

# Random Forest
rf_model=RandomForestClassifier(n_estimators=100)
rf_model.fit(trainX,trainY)
rf_predictions=rf_model.predict(testX)

Accuracy_score=accuracy_score(testY, rf_predictions)
Classification_report=classification_report(testY, rf_predictions)
F1=f1_score(testY, rf_predictions, average='macro')
Confusion_matrix=confusion_matrix(testY, rf_predictions)

# print("Random Forest Results:")
# print(f"Accuracy: {Accuracy_score}")
# print(f"F1 Score: {F1}")
# print("Classification Report:")
# print(Classification_report)
# print("Confusion Matrix:")
# print(Confusion_matrix)


#seraching for any duplicate in the dataset
duplicates=data.duplicated().sum()
# print(f"Number of duplicate rows in the dataset: {duplicates}")

from sklearn.model_selection import cross_val_score, StratifiedKFold

cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    rf_model,
    trainX,
    trainY,
    cv=cv,
    scoring='f1_macro'
)
# print(f"Cross-validated F1 scores: {cv_scores}")
# print(f"Mean F1 score: {cv_scores.mean()}")
# print(f"Standard deviation F1 score: {cv_scores.std()}")


rf_model=RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(trainX,trainY)
rf_predictions=rf_model.predict(testX)

Accuracy_score=accuracy_score(testY, rf_predictions)
Classification_report=classification_report(testY, rf_predictions)
F1=f1_score(testY, rf_predictions, average='macro')
Confusion_matrix=confusion_matrix(testY, rf_predictions)


print("Random Forest Results:")
print(f"Accuracy: {Accuracy_score}")
print(f"F1 Score: {F1}")
print("Classification Report:")
print(Classification_report)
print("Confusion Matrix:")
print(Confusion_matrix)

from sklearn.model_selection import GridSearchCV

rf_for_tuning = RandomForestClassifier(
    random_state=42,
    n_jobs=-1 #allows sklearn to use the cpu core
)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 20, 40],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search= GridSearchCV(
  estimator=rf_for_tuning, #model to tune
  param_grid= param_grid, #settings to test
  scoring='f1_macro',# choose the configuration with best macro F1
  cv=cv, #use the 5 fold stratifiied cross validation
  n_jobs=-1, 
  verbose=2 #display progresss
)