# BankModel

**Content**

The repo contains a package for development of the propensity model for credit cards based on dataset from Santander. Technically, the model is based on XGBoost technique. 
In order to find the best approach, H2O AutoML can be launched - dedicated implementation can be found in scripts/automl.py script. You need to have Java installed on your machine to use H2O AutoML. 

The dataset can be downloaded from [here](https://www.kaggle.com/competitions/santander-product-recommendation/data) - "train_ver2.csv.zip"

**Target definition**:

In a particular month (denoted as t) as a success ("1") we mark a customer who did not have a credit card three months before month t (it means, did not have a credit card in months t-3, t-2 and t-1), and have a credit card in a month t and one more month (t+1).
This is in order to avoid so-called "empty sale" - customers who bought a credit card, but did not use it in practice.

**How to launch the code?**

1. Create a new conda repository with the settings contained in **environment.yml**. Pull the repository.
2. Inside the project folder, create folder **data**. Inside folder data, create also folder **data_recommendation_engine**.
3. Download **train_ver2.csv.zip** from the link above and paste it in **data_recommendation_engine**. Unpack it here.
4. In **data** folder, create empty folder **trained_instances** - in this folder, serialized model will be saved.
5. Run **train.py** script and wait for the results

**Settings**

Global settings in **config.ini** file:

1. **feature_selection** - True/False - True means that **train.py** script will return results of feature selection, without any model. It is intended to manually choose important variables based on 5 algorithms: Information Value, Recursive Feature Elimination, Extratrees Classifier, Chi Square statistics and L1 regression. Also, VIF for multicollinearity is produced. It is intended to choose variables in expert-based approach
2. **opt_model** - True/False - If you also want to develop additional simplified model, where less important variables from the first one are excluded (based on feature_importance of initial xgboost model)
3. **garbage_model** - True/False - If True, additional analysis is launched where all variables from the step 1  (feature_selection) are used to double-chech whether any significant variable was excluded.
