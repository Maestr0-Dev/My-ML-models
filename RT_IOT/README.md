
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



Continuing my investigation on the features of this dataset, i tried to evaluate the relationship between potentially interesting categorical features and the "Attack_type". So i ran the following code:
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


Moving to numerical features, I wanted to understand how each features varied with each Attack_type. I started with the "flow_duration" and "fwd_pkts_tot" and "bwd_pkts_tot"

-------------------------------------------------------------
print(data.groupby('Attack_type')['flow_duration'].mean())
print(data.groupby('Attack_type')['fwd_pkts_tot'].mean())
---------------------------------------------------------------


the ouput showed that:  - Wipro_buld attacks have mean largly diffrent from that of other attacks accross these 3 features.
                        - Different attack types generate substantially different numbers of forward packets, making *fwd_pkts_tot* potentially useful for distinguishing attack types.

 i proceeded by running *print(data.groupby('Attack_type')['fwd_pkts_tot'].describe())* which gives some informations on *fwd_pkts_tot*  for each attack type. the out put:

 -----------------------------------------------------------------------------------------------
                               count       mean         std  min   25%   50%    75%     max
Attack_type                                                                               
ARP_poisioning               7750.0   8.296903   40.537051  0.0   1.0   2.0   6.00  2166.0
DDOS_Slowloris                534.0   6.041199    1.642022  1.0   4.0   6.0   7.75    11.0
DOS_SYN_Hping               94659.0   1.000000    0.000000  1.0   1.0   1.0   1.00     1.0
MQTT_Publish                 4146.0  10.171973   25.851223  1.0   9.0  10.0  10.00  1661.0
Metasploit_Brute_Force_SSH     37.0  12.216216    6.320756  1.0  14.0  14.0  15.00    33.0
NMAP_FIN_SCAN                  28.0   1.214286    0.956736  1.0   1.0   1.0   1.00     6.0
NMAP_OS_DETECTION            2000.0   1.000000    0.000000  1.0   1.0   1.0   1.00     1.0
NMAP_TCP_scan                1002.0   1.009980    0.140998  1.0   1.0   1.0   1.00     3.0
NMAP_UDP_SCAN                2590.0   2.069884   24.809871  0.0   1.0   1.0   1.00   903.0
NMAP_XMAS_TREE_SCAN          2010.0   1.003483    0.135665  1.0   1.0   1.0   1.00     7.0
Thing_Speak                  8108.0   5.392452    5.099881  0.0   2.0   2.0   7.00   130.0
Wipro_bulb                    253.0  80.529644  407.144582  0.0   1.0  10.0  16.00  4345.0
-----------------------------------------------------------------------------------------------------

 From this we can see that, fwd_pkts_tot varies substantially across some attack types, while several attack types have similar distributions. Therefore, the feature may be useful for distinguishing certain classes, but it is unlikely to be sufficient on its own for multiclass classification.


 After analysing the data, it time to prepare it before train the models
  fist thing to do was to drop the useless columns; "Unnamed: 0" and "bwd_URG_flag_count", then separate the taregt label form the features . Which leaves a total of 81 features to fit the models.

I then encoded the categorical data such that they can be fitted to machine learning classifiers as numerical values. the two columns( proto and service) containing the categorical data after encoding produced the new columns; *'proto_icmp', 'proto_tcp', 'proto_udp', 'service_-', 'service_dhcp', 'service_dns', 'service_http', 'service_irc', 'service_mqtt', 'service_ntp', 'service_radius', 'service_ssh', 'service_ssl'*

Here is how i proceeded with the different models

*Logic regression*

 I had to start by standardizing the data for every features. Why? Because the features operate on very different scales. Without standardization, a feature with large numerical values can disproportionately influence Logistic Regression.

 But the standardization had to be done only on the features which did not go throught the encoding phase. So i jad to drop those encoded features, standardize the rest which made them into numpy arrays, transform it back to Dataframes and concatenate back with the encoded features to recreate the training and test data.

 










