# Real-Time Network Attack Detection Using Machine Learning

## 1. Introduction

For my first machine learning project, I decided to work on network security. The goal was to use machine learning to classify network traffic into different traffic or attack categories.

I used the RT-IoT2022 dataset from the UCI Machine Learning Repository. The dataset contains network traffic collected from IoT devices and includes different types of normal and malicious traffic.

The main target variable is `Attack_type`.

-----------------------------------------

# 2. Objective of the Project

The objective of this project is to build a machine learning model that can classify network traffic into its corresponding `Attack_type` using features extracted from IoT network traffic.

I also wanted to:

* understand the dataset before training a model
* compare different classification algorithms
* see how class imbalance affects the results
* tune the best-performing model
* identify which features were most useful for the final model
* save the final model so it can be used later for predictions

-------------

# 3. Dataset

I used the RT-IoT2022 dataset from the UCI Machine Learning Repository.
Dataset source:
[https://archive.ics.uci.edu/dataset/942/rt-iot2022](https://archive.ics.uci.edu/dataset/942/rt-iot2022)

The dataset contains:
* 123,117 rows
* 85 columns in the CSV file
* network traffic features extracted from IoT network traffic
* the target variable `Attack_type`

The target contains 12 classes in the version of the dataset I used:

1. `ARP_poisioning`
2. `DDOS_Slowloris`
3. `DOS_SYN_Hping`
4. `MQTT_Publish`
5. `Metasploit_Brute_Force_SSH`
6. `NMAP_FIN_SCAN`
7. `NMAP_OS_DETECTION`
8. `NMAP_TCP_scan`
9. `NMAP_UDP_SCAN`
10. `NMAP_XMAS_TREE_SCAN`
11. `Thing_Speak`
12. `Wipro_bulb`

The classes are highly imbalanced. For example, `DOS_SYN_Hping` represents about 76.9% of the dataset, while `Metasploit_Brute_Force_SSH` represents only about 0.03%.
This became important later when evaluating the models.

------------------------------

# 4. Initial Data Inspection

I first loaded the dataset and checked its general structure.

data = pd.read_csv('./Datasets/RT_IOT2022')

print(data.shape)
print(data.head())
print(data.info())

The dataset contained 123,117 observations and 85 columns.
I also checked for missing values:

data.isnull().sum()

There were no missing values that needed to be handled.

-____________________________________________

# 5. Checking the Features
I cheked the number of unique values in each column:

data.nunique().sort_values()

This helped me identify columns with very few unique values.
One of the columns, `bwd_URG_flag_count`, had only one unique value across the entire dataset. Since it had the same value for every observation, it could not provide useful information for classification.
There was also an `Unnamed: 0` column, which appeared to be an index-like column rather than a meaningful network feature.

I therefore removed both:
______________________________________________
polished_data = data.drop(
    ['Unnamed: 0', 'bwd_URG_flag_count'],
    axis=1
)
_________________________________________________

I did not want the model to learn from an artificial index or a constant feature.

-----------------------------------------------------

# 6. Investigating the Target Classes

I checked the distribution of `Attack_type` because I expected the classes to be imbalanced.

data['Attack_type'].value_counts()

The largest class was `DOS_SYN_Hping`
with approximately 76.9% of all observations.
Some classes had very few observations, especially:

* `Metasploit_Brute_Force_SSH`
* `NMAP_FIN_SCAN`
* `Wipro_bulb`

This meant that accuracy alone would not be enough to evaluate the models. A model could achieve high accuracy simply by performing very well on the largest class.

For this reason, I focused particularly on Macro F1, which gives each class equal importance.

--------------------------------------------

# 7. Investigating Categorical Features

Two columns were categorical:

* `proto`
* `service`

I examined their relationship with `Attack_type`.
For example, some services and protocols were strongly associated with certain classes.

Some observations included:

* most `MQTT_Publish` traffic used the MQTT service
* `Wipro_bulb` was strongly associated with IRC traffic
* `Thing_Speak` had strong associations with ICMP and HTTP
* `NMAP_OS_DETECTION` was strongly associated with TCP

At first, I considered whether these relationships could represent target leakage.

However, strong association by itself does not mean leakage. These are actual network traffic characteristics that would be available when classifying a network flow.
Therefore, I kept `proto` and `service` as features.

------------------------------------

# 8. Exploring Numerical Features

I also investigated numerical features to understand whether different traffic categories had different patterns.

One feature I looked at was:

`fwd_pkts_tot`

I compared its statistics across the different classes.

For example:

* `DOS_SYN_Hping` had a mean of approximately 1 forward packet
* `MQTT_Publish` had a mean of approximately 10.17
* `Wipro_bulb` had a much higher mean of approximately 80.53
* `Thing_Speak` had a mean of approximately 5.39

This suggested that packet-related features could contain useful information for distinguishing between traffic categories.

I also investigated features such as:

* `flow_duration`
* packet counts
* payload statistics
* inter-arrival times

---------------------

# 9. Preparing the Data

I separated the input features from the target:
_________________________________________________
X = polished_data.drop(['Attack_type'], axis=1)
y = polished_data['Attack_type']
_______________________________________________


Since `proto` and `service` were categorical, I converted them into numerical features using one-hot encoding:
__________________________________________
X = pd.get_dummies(
    X,
    columns=["proto", "service"],
    dtype=int
)
__________________________________________

I then divided the data into training and testing sets.
__________________________________________________
trainX, testX, trainY, testY = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
__________________________________________________________

I used an 80/20 split.

I also used `stratify=y` so that the class distribution would be maintained approximately in both sets.

------------------------------------------

# 10. Logistic Regression

I started with Logistic Regression as a baseline model.
Because Logistic Regression is sensitive to feature scale, I standardized the numerical features using `StandardScaler`.

I did not standardize the one-hot encoded categorical variables.

The model achieved approximately:

* Accuracy: 99.14%
* Macro F1: 0.92
* Weighted F1: 0.99

Although the overall accuracy was high, performance was weaker on some minority classes.

For example:

* `DDOS_Slowloris`: F1 ≈ 0.88
* `Wipro_bulb`: F1 ≈ 0.72
* `NMAP_FIN_SCAN`: F1 ≈ 0.77
* `Metasploit_Brute_Force_SSH`: F1 ≈ 0.86

This showed why looking only at accuracy would be misleading.

---

# 11. Decision Tree

I then tested a Decision Tree Classifier.

DecisionTreeClassifier()

The results were:

* Accuracy: 99.79%
* Macro F1: 0.969

The Decision Tree performed better than Logistic Regression.
In particular, performance on some of the more difficult classes improved.

For example, `Wipro_bulb` recall increased from about 0.59 with Logistic Regression to about 0.82 with the Decision Tree.
This suggested that nonlinear relationships between the features were useful for this dataset.

---

# 12. Random Forest

I then tested a Random Forest Classifier.My initial Random Forest used 100 trees.
The results were:

* Accuracy: 99.85%
* Macro F1: 0.977

This was the strongest baseline model.
The Random Forest performed very well across most classes, including the minority classes.

However, some classes still had very small test support. For example:

* `Metasploit_Brute_Force_SSH`: 7 test samples
* `NMAP_FIN_SCAN`: 6 test samples

Therefore, the performance numbers for these classes should not be treated as highly reliable conclusions.

---

# 13. Model Comparison

The three initial models gave the following results:

| Model               | Accuracy | Macro F1 | Weighted F1 |
| ------------------- | -------: | -------: | ----------: |
| Logistic Regression |   99.14% |     0.92 |        0.99 |
| Decision Tree       |   99.79% |    0.969 |       ~1.00 |
| Random Forest       |   99.85% |    0.977 |       ~1.00 |

Random Forest was the best of the three based on Macro F1 and accuracy.
I therefore continued with Random Forest.

-----------------------------------------------

# 14. Cross-Validation

Since a single train/test split can sometimes give misleading results, I also used 5-fold stratified cross-validation.
________________________________________________
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    rf_model,
    trainX,
    trainY,
    cv=cv,
    scoring='f1_macro'
)
_________________________________________________


The Macro F1 scores across the folds were approximately:

0.96497
0.96865
0.96093
0.96707
0.96592

The mean was: 0.96551

The results were relatively close across the five folds, suggesting that the model's performance was reasonably consistent across different subsets of the training data.

However, cross-validation does not prove that there is no leakage or bias in the dataset.

--------------------------------------------------

# 15. Testing Class Weighting
Because the dataset was highly imbalanced, I also tested a class-weighted Random Forest.

rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)

The results were:

* Accuracy: 99.83%
* Macro F1: 0.975
This was slightly worse than the original Random Forest.
Therefore, using `class_weight='balanced'` did not improve the model in this experiment.

I decided to keep the normal Random Forest.

---=-------------------------------

# 16. Hyperparameter Tuning

I then tried to improve the Random Forest using `GridSearchCV`.

For the final tuning experiment, I tested:
________________________________________________________-
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 20]
}

I used Macro F1 as the scoring metric because of the class imbalance.

rf_for_tuning = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

grid_search = GridSearchCV(
    estimator=rf_for_tuning,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=cv,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(trainX, trainY)

The best parameters were:

n_estimators = 200
max_depth = 20
______________________________________________________________
The best cross-validation Macro F1 was approximately: 0.9645

The improvement from tuning was relatively small, which is not surprising because the baseline Random Forest was already performing very well.

---

# 17. Final Model

The final model was the tuned Random Forest:

Random Forest
n_estimators = 200
max_depth = 20
random_state = 42

Its final test results were:

* Accuracy: 99.85%
* Macro F1: 0.976
* Weighted F1: 0.998

The classification report showed very strong performance across most classes.

The most difficult classes were still mainly the classes with fewer examples.

For example:

| Class                      |   F1 |
| -------------------------- | ---: |
| ARP_poisioning             | 0.99 |
| DDOS_Slowloris             | 1.00 |
| DOS_SYN_Hping              | 1.00 |
| MQTT_Publish               | 1.00 |
| Metasploit_Brute_Force_SSH | 0.88 |
| NMAP_FIN_SCAN              | 0.91 |
| NMAP_OS_DETECTION          | 1.00 |
| NMAP_TCP_scan              | 1.00 |
| NMAP_UDP_SCAN              | 1.00 |
| NMAP_XMAS_TREE_SCAN        | 1.00 |
| Thing_Speak                | 0.99 |
| Wipro_bulb                 | 0.96 |

The overall performance was very high, but the results for classes with only a few test examples should be interpreted carefully.

---

# 18. Confusion Matrix

I also generated a confusion matrix for the final Random Forest.

The confusion matrix showed that most observations were correctly classified.

Most of the remaining errors involved classes that were relatively similar in their network characteristics.

For example, some `Thing_Speak` observations were classified as `ARP_poisioning`, while some `Wipro_bulb` observations were classified as other traffic categories.

Overall, there were very few misclassifications compared with the total number of test observations.

---

# 19. Feature Importance

After training the final Random Forest, I examined its feature importance values.

The most important features included:

| Feature                | Importance |
| ---------------------- | ---------: |
| `fwd_pkts_payload.avg` |     0.0789 |
| `id.resp_p`            |     0.0742 |
| `fwd_pkts_payload.min` |     0.0560 |
| `fwd_subflow_bytes`    |     0.0485 |
| `fwd_pkts_payload.max` |     0.0468 |
| `fwd_pkts_payload.tot` |     0.0404 |
| `service_-`            |     0.0337 |
| `active.avg`           |     0.0309 |
| `flow_duration`        |     0.0290 |
| `flow_iat.min`         |     0.0259 |

Other important features included flow inter-arrival times, payload statistics and activity measurements.

This suggests that characteristics such as packet payloads, destination ports, flow duration, timing and network service information were important for the Random Forest's decisions.

These feature importance values show which features were useful for the tree splits; they do not mean that the features directly cause a particular attack category.

---

# 20. Saving the Model

Once I had selected the final model, I saved it using `joblib`:
_______________________________________
joblib.dump(
    best_rf,
    'random_forest_attack_predictor.pkl'
)
_____________________________________


This means I do not need to retrain the Random Forest every time I want to use it.Later, I can load the saved model with:

____________________________________________-
model = joblib.load(
    'random_forest_attack_predictor.pkl'
)
________________________________________________

The new data would still need to go through the same preprocessing steps used during training before being passed to the model.

---------------------------------------------------

# 21. Duplicate Check
I also checked whether the original dataset contained duplicate rows:
print("Duplicate rows:", data.duplicated().sum())

Result: 0

This check is important because if identical observations occur in both the training and testing sets, the model may appear to perform better than it would on genuinely new traffic.

---

# 22. Final Model Comparison

My overall model progression was:

| Model               |   Accuracy |  Macro F1 |
| -------------- ---  | ---------: | --------: |
| Logistic Regression |     99.14% |     0.920 |
| Decision Tree       |      99.79% |     0.969 |
| Random Forest       |     99.85% |     0.977 |
| Tuned Random Forest | 99.85%    | 0.976 |

The tuning did not produce a major improvement over the original Random Forest.

This was useful because it showed me that more tuning does not automatically mean a much better model. The original Random Forest was already very strong.

---------------------------------------

# 23. Limitations
There are several limitations to this project.

## 1. Strong class imbalance
`DOS_SYN_Hping` makes up around 77% of the dataset.
This is why I focused on Macro F1 instead of relying only on accuracy.

## 2. Very small classes
Some classes have extremely few observations. For example, the final test set contained only:

* 7 `Metasploit_Brute_Force_SSH`
* 6 `NMAP_FIN_SCAN`

It is difficult to make strong conclusions about model performance on these classes with such small samples.

## 3. Random train/test split
The model was evaluated using a random stratified split.
In a real network environment, traffic could come from completely different devices, networks or time periods. A future version of the project could test the model on a more genuinely unseen source of traffic.

## 4. Test-set model selection
I experimented with more than one hyperparameter grid and compared their final test results. Strictly speaking, the test set should ideally be used only once for the final evaluation.

A better future approach would be to choose the model entirely using cross-validation and then evaluate it once on the untouched test set.

## 5. Preprocessing could be made more rigorous

I performed one-hot encoding before the train/test split. This did not use the target variable, so it is not target leakage, but a more rigorous implementation would fit the preprocessing steps only on the training data and then apply them to the test data.

----------------------------------------

# 24. Conclusion
In this project, I built a machine learning system for classifying IoT network traffic using the RT-IoT2022 dataset.

I started by exploring the data, checking missing values, investigating class imbalance and examining the relationships between network features and the target.

I compared three models:
* Logistic Regression
* Decision Tree
* Random Forest

Random Forest performed the best. I then tested class weighting and hyperparameter tuning, but these only produced small changes.

The final tuned Random Forest achieved approximately:

99.85% accuracy
0.976 Macro F1
0.998 Weighted F1

The model performed very well across most of the traffic categories. However, the results for the rarest classes need to be interpreted carefully because there are very few examples of them.

I also examined feature importance and found that payload statistics, destination port, flow timing, activity and service information were among the most important features.

Overall, this project helped me understand the complete machine learning workflow, from data exploration and preprocessing to model comparison, evaluation, tuning and saving a model for future use.
