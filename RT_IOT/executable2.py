import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LogicRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix, f1_score

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


numerical=trainX.drop(['proto_icmp', 'proto_tcp', 'proto_udp', 'service_-', 'service_dhcp', 'service_dns', 'service_http', 'service_irc', 'service_mqtt', 'service_ntp', 'service_radius', 'service_ssh', 'service_ssl'],axis=1)
scaler.fit(numerical)
num=scaler.transform(numerical)
num=pd.DataFrame(num, columns=numerical.columns, index=numerical.index)
trainX=pd.concat([num, trainX[['proto_icmp', 'proto_tcp', 'proto_udp', 'service_-', 'service_dhcp', 'service_dns', 'service_http', 'service_irc', 'service_mqtt', 'service_ntp', 'service_radius', 'service_ssh', 'service_ssl']]], axis=1)

numerical=testX.drop(['proto_icmp', 'proto_tcp', 'proto_udp', 'service_-', 'service_dhcp', 'service_dns', 'service_http', 'service_irc', 'service_mqtt', 'service_ntp', 'service_radius', 'service_ssh', 'service_ssl'],axis=1)
num=scaler.transform(numerical)
num=pd.DataFrame(num, columns=numerical.columns, index=numerical.index)
testX=pd.concat([num, testX[['proto_icmp', 'proto_tcp', 'proto_udp', 'service_-', 'service_dhcp', 'service_dns', 'service_http', 'service_irc', 'service_mqtt', 'service_ntp', 'service_radius', 'service_ssh', 'service_ssl']]], axis=1)


model=LogicRegressor()
model.fit(trainX,trainY)

predictions=model.predict(testX)

for i in range(1,30):
    print(f"Predicted: {predictions[i]}, Actual: {testY.iloc[i]}")

Accuracy_score=[accuracy_score(testY, predictions)]
Classification_report=[classification_report(testY, predictions)]
F1=[f1_score(testY, predictions, average='weighted')]
Confusion_matrix=[confusion_matrix(testY, predictions)] 




