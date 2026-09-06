import pandas as pd
# from config import data_path

data=pd.read_csv('./Datasets/RT_IOT2022')

# print(data.head())
print('data columns: \n',data.columns)

# print('Info: \n',data.info())

# print('Describe: \n',data.describe())  
      
# print("describing the label collumn \n",data['Attack_type'].describe())

# X=data.drop(['Attack_type'],axis=1)
# print(data['Attack_type'].value_counts(normalize=True))
# print(data.duplicated().sum())
# print(data.nunique().sort_values())


# unique_columns=data.columns[data.nunique() == 1]
# print(unique_columns)

# print(data['Unnamed: 0'].nunique())

# print(data['Unnamed: 0'].head(20))
# print(data['Unnamed: 0'].tail(20))
# zero_count=data['Unnamed: 0'].value_counts().get(0,0)
# print("Count of zeros in 'Unnamed: 0':", zero_count)

# print(data["id.orig_p"].describe())
# print(data["id.resp_p"].describe())
# print(data.nunique().sort_values().head(20))

# print(data["id.resp_p"].value_counts())

# print(pd.crosstab(data["proto"], data["Attack_type"], normalize="index"))
# print(pd.crosstab(data["service"], data["Attack_type"], normalize="index"))

# print(pd.crosstab(data["Attack_type"], data["proto"], normalize="index"))
# print(pd.crosstab(data["Attack_type"], data["service"], normalize="index"))

# Analysing the numerical features

# num1=data.select_dtypes(include=['float64','int64']).columns

# num=num1.drop(['Unnamed: 0', 'bwd_URG_flag_count'])

# av_per_num=data.groupby("Attack_type")[num].mean()
# print(av_per_num)

# print( data.groupby('Attack_type')['flow_duration'].mean())
# print(data.groupby('Attack_type')['fwd_pkts_tot'].mean())
# print(data.groupby("Attack_type")["bwd_pkts_tot"].mean())

# print(data.groupby('Attack_type')['fwd_pkts_tot'].describe())
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
sns.boxplot(data=data, x="Attack_type", y="fwd_pkts_tot")
plt.xticks(rotation=90)
plt.show()
