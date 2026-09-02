
*Project: Real-Time Network Attack Detection*
objective of the project:
-----
-----
process:

The dataset documentation on ___ said there are no missing values but i still needed to verify. 
i ran *data.pd.isnull().sum()* , and with no surprise, the data was ok with no missing values.

i started by bringing out the list of all the features in the dataset and based on their characteristics, bring out which ones will be used as target when trainign the machine learning models.
 From this, the target column is the 'Attack_type' column.

Now i have to evaluate which columns are categorical.
 ____
 ____
 
after runnig *print(data['Attack_type'].value_counts(normalize=True))* i noticed an enormose imbalance in the dataset. "DOS_SYN_Hiping"attacks alone constitutes roughly 77% of the dataset, while ___ constitues about 0.02% of the dataset. 
-------------------------------------------
       Attack_type                          |
DOS_SYN_Hping                 0.768854      |
Thing_Speak                   0.065856      |
ARP_poisioning                0.062948      |
MQTT_Publish                  0.033675      |
NMAP_UDP_SCAN                 0.021037
NMAP_XMAS_TREE_SCAN           0.016326
NMAP_OS_DETECTION             0.016245
NMAP_TCP_scan                 0.008139
DDOS_Slowloris                0.004337
Wipro_bulb                    0.002055
Metasploit_Brute_Force_SSH    0.000301
NMAP_FIN_SCAN                 0.000227
Name: proportion, dtype: float64 
--------------------------------------------

    This could cause the future models to have a high accuracy score meanwhile it actually failed at predicting several types of attacks and could only predict 'DOS_SYN_Hiping'.


while analysin the data using the command *data.nunique().sort_values()* i noticed some interesting characteristics about some columns.
    1. *bwd_URG_flag_count* : this column has only one values through out all the 123117 records. this is a signal that the column will may not be used durring the classification phase, since it does not provides any distinction between classes. To find similar columns, i ran the following code in otehr to verify if there's any other column with 1 value:
    ---------------------------------------------------
    unique_columns=data.columns[data.nunique() == 1]
    print(unique_columns)
    ---------------------------------------------------
    The output showed that this is the only columns with single value.

    2. Unnamed: 0" : his column initially appeared to be an index because its first values corresponded to the dataframe index. However, further inspection showed that the values do not continue sequentially throughout the dataset. I also found from the code below that the value 0 occurs 12 times, while other values are repeated, suggesting that the column follows a recurring sequence rather than representing a unique identifier. 
    
    --------------------------------------------------------
    zero_count=data['Unnamed: 0'].value_counts().get(0,0)
    print("Count of zeros in 'Unnamed: 0':", zero_count)
    --------------------------------------------------------

    The ouput was 12. This means that there is a total of 12 sequences, where each starts with a 0 to a certain integer X. I did not take time to find the X because i already had the answer to my question (Is "Unnamed: 0" a necessary column which can be used during classification? No)

    Since this column does not appear to represent an intrinsic characteristic of the network traffic, I consider it unsuitable as a predictive feature and will exclude it from the classification stage.



Continuing my investigation on the features of this dataset, i tried to evaluate the relationship between potentially interesting features and the "Attack_type". So i ran the following code:
----------------------------------------------------------------------------
print(pd.crosstab(data["proto"], data["Attack_type"], normalize="index"))
print(pd.crosstab(data["service"], data["Attack_type"], normalize="index"))
----------------------------------------------------------------------------

From the out put, i noticed a strong relationship between the features. For example;
    On the Proto X Attack_type table, i noticed that *85.7% of all TCPs observations where DOS_SYN_Hping*and on the service X Attack_type table, *100% of all IRCs where Wipro_bulb*. such high.
    Other examples: 
    -78.9% of observations where protocol= "icmp" are classified as Thing_speak
    -76.9% of observations where service="http" are Thing_speak

Such highly predictive information lead me to think that these features (proto and service) could create a *Leakage*

To have a better eye on this, I switche the point of view. Instead of finding what attack occurs given protocol/service, i found what protocol/service occurs given the Attack_type.

----------------------------------------------------------------------------
print(pd.crosstab(data["Attack_type"], data["proto"], normalize="index"))
print(pd.crosstab(data["Attack_type"], data["service"], normalize="index"))
----------------------------------------------------------------------------

From this, i had the following observation;
- 99.6% of MQT_PUBLISH attacks use mqt sevrice
- 100% of NMAP_OS_DETECTION attacks use tcp protocol

In conclusion to this, proto and service show strong associations with Attack_type, making them potentially valuable predictive features. However, strong association alone does not prove target leakage.












