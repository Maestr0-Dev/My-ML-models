
*Project: Real-Time Network Attack Detection*
objective of the project:
-----
-----
                                            **PROCESS**
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

Here is how i proceded with the different models

**MODEL TRAINING ,TESTING AND SELECTION**

*1. Logic regression*

 I had to start by standardizing the data for every features. Why? Because the features operate on very different scales. Without standardization, a feature with large numerical values can disproportionately influence Logistic Regression.

 But the standardization had to be done only on the features which did not go throught the encoding phase. So i had to drop those encoded features, standardize the rest which made them into numpy arrays, transform it back to Dataframes and concatenate back with the encoded features to recreate the training and test data.

Then came the implementation of the Logistic Regression model;

-----------------------------------------
*model=LogicRegressor()*
*model.fit(trainX,trainY)*

*predictions=model.predict(testX)*
------------------------------------------

But i could not just stay at prediction, i had to evaluate the models throught differnt methods;
accuracy score, confussion metrix, F1 and classification report.

-------------------------------------------------------------------
*Accuracy_score=[accuracy_score(testY, predictions)]*
*Classification_report=[classification_report(testY, predictions)]*
*F1=[f1_score(testY, predictions, average='weighted')]*
*Confusion_matrix=[confusion_matrix(testY, predictions)]*

---------------------------------------------------------------------

i had the following results:
Logistic Regression Results:
Accuracy: 0.9913905133203379
F1 Score: 0.9912696193237263
Classification Report:
                            precision    recall  f1-score   support

            ARP_poisioning       0.95      0.94      0.95      1550
            DDOS_Slowloris       0.98      0.80      0.88       107
             DOS_SYN_Hping       1.00      1.00      1.00     18932
              MQTT_Publish       1.00      1.00      1.00       829
Metasploit_Brute_Force_SSH       0.86      0.86      0.86         7
             NMAP_FIN_SCAN       0.71      0.83      0.77         6
         NMAP_OS_DETECTION       0.99      1.00      1.00       400
             NMAP_TCP_scan       1.00      1.00      1.00       200
             NMAP_UDP_SCAN       0.96      0.98      0.97       518
       NMAP_XMAS_TREE_SCAN       1.00      1.00      1.00       402
               Thing_Speak       0.94      0.96      0.95      1622
                Wipro_bulb       0.94      0.59      0.72        51

                  accuracy                           0.99     24624
                 macro avg       0.94      0.91      0.92     24624
              weighted avg       0.99      0.99      0.99     24624

Confusion Matrix:
[[ 1456     1     0     4     0     0     0     1     0     0    87     1]
 [    1    86     0     0     0     0     0     0    20     0     0     0]
 [    0     0 18932     0     0     0     0     0     0     0     0     0]
 [    2     0     0   827     0     0     0     0     0     0     0     0]
 [    1     0     0     0     6     0     0     0     0     0     0     0]
 [    1     0     0     0     0     5     0     0     0     0     0     0]
 [    0     0     0     0     0     0   400     0     0     0     0     0]
 [    0     0     0     0     0     0     0   200     0     0     0     0]
 [    7     1     0     0     0     0     0     0   509     0     1     0]
 [    2     0     0     0     0     0     0     0     0   400     0     0]
 [   55     0     0     0     1     1     0     0     3     0  1561     1]
 [    1     0     0     0     0     1     3     0     0     0    16    30]]



From these measurements, 
>>>F1 macro=0.92, signifying that the model performs pretty overall well but classes remain more difficult to detect than others.

>>> Wipro_bulb Recall = 0.59; That means the model correctly detects only about 59% of actual Wipro_bulb attacks.
There are 51 Wipro_bulb examples:
30 correctly identified
16 classified as Thing_Speak
3 classified as NMAP_OS_DETECTION
1 as ARP_poisioning
1 as NMAP_FIN_SCAN

So the model is particularly confusing Wipro_bulb and Thing_Speak

>>>DDOS_Slowloris is another weakness. There are 107 actual examples but the model correctly identifies 86 / 107
But 20 are classified as NMAP_UDP_SCAN. So the model is confused between NMAP_UDP_SCAN and DDOS_Slowloris.

>>>Classes such as NMAP_FIN_SCAN and Metasploit_Brute_Force_SSH contain  way too litle samples to be a ble to draw confident conclusions

>>> the 99.1% accuracy combined with 92% macro F1 tells us that The model isn't just exploiting the majority class. It is genuinely performing well across most classes, but there are some minority-class weaknesses.

*conclusions for the logistic regression model*
The Logistic Regression achieved 99.14% accuracy and a macro F1-score of 0.92. Performance was excellent for most attack classes, with perfect classification for several NMAP and traffic categories. However, Wipro_bulb and DDOS_Slowloris showed lower recall, primarily due to confusion with Thing_Speak and NMAP_UDP_SCAN respectively. Extremely rare classes such as NMAP_FIN_SCAN and Metasploit_Brute_Force_SSH contain too few test samples for reliable performance assessment.


*2. Decision Tree*

Decision Tree Results:
Accuracy: 0.9979694606887589
F1 Score: 0.9688486346839085
Classification Report:
                            precision    recall  f1-score   support

            ARP_poisioning       0.99      0.99      0.99      1550
            DDOS_Slowloris       0.99      1.00      1.00       107
             DOS_SYN_Hping       1.00      1.00      1.00     18932
              MQTT_Publish       1.00      1.00      1.00       829
Metasploit_Brute_Force_SSH       0.78      1.00      0.88         7
             NMAP_FIN_SCAN       1.00      0.83      0.91         6
         NMAP_OS_DETECTION       1.00      1.00      1.00       400
             NMAP_TCP_scan       1.00      1.00      1.00       200
             NMAP_UDP_SCAN       0.99      0.99      0.99       518
       NMAP_XMAS_TREE_SCAN       1.00      1.00      1.00       402
               Thing_Speak       0.99      0.99      0.99      1622
                Wipro_bulb       0.95      0.82      0.88        51

                  accuracy                           1.00     24624
                 macro avg       0.97      0.97      0.97     24624
              weighted avg       1.00      1.00      1.00     24624

Confusion Matrix:
[[ 1533     0     0     0     1     0     0     0     2     0    14     0]
 [    0   107     0     0     0     0     0     0     0     0     0     0]
 [    0     0 18932     0     0     0     0     0     0     0     0     0]
 [    1     1     0   827     0     0     0     0     0     0     0     0]
 [    0     0     0     0     7     0     0     0     0     0     0     0]
 [    0     0     0     0     0     5     0     0     0     1     0     0]
 [    0     0     0     0     0     0   400     0     0     0     0     0]
 [    0     0     0     0     0     0     0   200     0     0     0     0]
 [    1     0     0     0     1     0     0     0   514     0     1     1]
 [    0     0     0     0     0     0     0     0     2   400     0     0]
 [   12     0     0     0     0     0     1     0     1     0  1607     1]
 [    0     0     0     2     0     0     0     0     0     0     7    42]]


This model performed arguiaby better than the logistic regression model.
with and accuracy score : 99.79%
>>>F1 macro: 0.96 , has significantly increased from that of the logistic regression model;0.92. signaling an improvement

>>> Wipro_bulb has recall=0.82, which is a big improvement form the 0.59 of the logistic regression model.
45 correctly identified,
4 classified as Thing_Speak, 
2 classified as MQTT_Publish
the model has little difficulties distinguishing between Wipro and classified.
>>>ARP_poisioning went from F1: 0.95 to 0.99
>>>Thing_Speak F1: 0.95 to 0.99

So the tree is clearly capturing nonlinear relationships that Logistic Regression wasn't capturing as well.

form the result, it is clear that *Wipro_bulb* is the most challenging class to predict.
Lets see what the Random Forrest classifier shows.

*3. Random Forrest*
Random Forest Results:
Accuracy: 0.9985380116959064
F1 Score: 0.9768259740066153
Classification Report:
                            precision    recall  f1-score   support

            ARP_poisioning       0.99      0.99      0.99      1550
            DDOS_Slowloris       1.00      0.99      1.00       107
             DOS_SYN_Hping       1.00      1.00      1.00     18932
              MQTT_Publish       1.00      1.00      1.00       829
Metasploit_Brute_Force_SSH       0.78      1.00      0.88         7
             NMAP_FIN_SCAN       1.00      0.83      0.91         6
         NMAP_OS_DETECTION       1.00      1.00      1.00       400
             NMAP_TCP_scan       1.00      1.00      1.00       200
             NMAP_UDP_SCAN       0.99      0.99      0.99       518
       NMAP_XMAS_TREE_SCAN       1.00      1.00      1.00       402
               Thing_Speak       0.99      0.99      0.99      1622
                Wipro_bulb       0.98      0.96      0.97        51

                  accuracy                           1.00     24624
                 macro avg       0.98      0.98      0.98     24624
              weighted avg       1.00      1.00      1.00     24624

Confusion Matrix:
[[ 1540     0     0     0     1     0     0     0     0     0     9     0]
 [    0   106     0     0     0     0     0     0     1     0     0     0]
 [    0     0 18932     0     0     0     0     0     0     0     0     0]
 [    2     0     0   827     0     0     0     0     0     0     0     0]
 [    0     0     0     0     7     0     0     0     0     0     0     0]
 [    0     0     0     0     0     5     0     0     0     1     0     0]
 [    0     0     0     0     0     0   400     0     0     0     0     0]
 [    0     0     0     0     0     0     0   200     0     0     0     0]
 [    2     0     0     0     1     0     0     0   515     0     0     0]
 [    1     0     0     0     0     0     0     0     1   400     0     0]
 [   13     0     0     0     0     0     0     0     1     0  1607     1]
 [    1     0     0     0     0     0     0     0     0     0     1    49]]


This is the stronggest of all so far.


| Model               |   Accuracy | Macro F1 | Weighted F1 |
| ------------------- | ---------: | ---------| ----------- |
| Logistic Regression |     99.14% |     0.92 |        0.99 |
| Decision Tree       |     99.79% |    0.96 |        1.00 |
| **Random Forest**   | **99.85%** | **0.97** |    **1.00** |


The improvement from Logistic Regression to Decision tree is greater than that from Decision tree to Random Forest.
But before concluding, we need to be sure that the models actually kearned the data, and not just cramming it. To verify that, i used a *5-fold stratified cross-validation of the Random Forest using macro F1*. This ecentailly means that, i will split the training data into 5groups while maintaining the proportions of each class, train the random forest 5 different times, testing each group different times and evaluating how well the model did each time using macro F1.

and the result was this: 
*Cross-validated F1 scores: [0.96496918 0.96864517 0.96092697 0.96706669 0.96592037]*
*Mean F1 score: 0.9655056750707878*


The mean value of the cross validated f1 score is very close to the actual F1 score of the model, which proves that the Random Forest model is really predicting attack types without cramming or halucinating.

The dataset is heavily imbalanced and majority classes dominate on minorities such as:

Metasploit_Brute_Force_SSH *>>>* only 7 test samples
NMAP_FIN_SCAN *>>>* only 6 test samples
Wipro_bulb *>>>* 51 test samples
DDOS_Slowloris *>>>* 107 test samples

May class weighting will improve minority-class performance. So I tried a class-balanced random-forest to compare the result.
The Class-Balanced random-forest gave the following results:

Random Forest Results:
Accuracy: 0.9983349577647823
F1 Score: 0.9745608400468327
Classification Report:
                            precision    recall  f1-score   support

            ARP_poisioning       0.99      0.99      0.99      1550
            DDOS_Slowloris       1.00      0.96      0.98       107
             DOS_SYN_Hping       1.00      1.00      1.00     18932
              MQTT_Publish       1.00      1.00      1.00       829
Metasploit_Brute_Force_SSH       0.78      1.00      0.88         7
             NMAP_FIN_SCAN       1.00      0.83      0.91         6
         NMAP_OS_DETECTION       1.00      1.00      1.00       400
             NMAP_TCP_scan       1.00      1.00      1.00       200
             NMAP_UDP_SCAN       0.99      0.99      0.99       518
       NMAP_XMAS_TREE_SCAN       1.00      1.00      1.00       402
               Thing_Speak       0.99      0.99      0.99      1622
                Wipro_bulb       0.98      0.94      0.96        51

                  accuracy                           1.00     24624
                 macro avg       0.98      0.98      0.97     24624
              weighted avg       1.00      1.00      1.00     24624

Confusion Matrix:
[[ 1539     0     0     0     1     0     0     0     0     0    10     0]
 [    0   103     0     0     0     0     0     0     4     0     0     0]
 [    0     0 18932     0     0     0     0     0     0     0     0     0]
 [    2     0     0   827     0     0     0     0     0     0     0     0]
 [    0     0     0     0     7     0     0     0     0     0     0     0]
 [    0     0     0     0     0     5     0     0     0     0     1     0]
 [    0     0     0     0     0     0   400     0     0     0     0     0]
 [    0     0     0     0     0     0     0   200     0     0     0     0]
 [    2     0     0     0     1     0     0     0   515     0     0     0]
 [    2     0     0     0     0     0     0     0     0   400     0     0]
 [   13     0     0     0     0     0     0     0     1     0  1607     1]
 [    1     0     0     1     0     0     0     0     0     0     1    48]]


Class weighting did not meaningfully improve your overall macro F1. It slightly reduced accuracy:from 99.85% to 99.83% 
And the minority-class results changed only slightly.

the Recall and F1 scores of DDOS_Slowloris, Wipro_bulb, NMAP_FIN_SCAN instead got slightly worse. But they occupy a very tiny portion of the dataset. For this Random Forest configuration and this dataset split, class weighting does not provide a meaningful improvement.

At the end i decided o=to go with the Random forest model and proceed with hyperparameter tuning.

**HYPERPARAMETER TUNNING**

I will use GridSearchCV to test several Random Forest configurations automatically and select the one with the best macro F1.
 I used the following parameter grid:

  param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 20, 40],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

which contains 24 different configurations.

