# BankModel

**Content**

The repo contains a package for development of the propensity model for credit cards based on dataset from Santander. Technically, the model is based on XGBoost technique. 
In order to find the best approach, H2O AutoML can be launched - dedicated implementation can be found in scripts/automl.py script. You need to have Java installed on your machine to use H2O AutoML. 

The dataset can be downloaded from [here](https://www.kaggle.com/competitions/santander-product-recommendation/data) - "train_ver2.csv.zip"

To format the code, **black** library was used.

IMPORTANT NOTICE: IV is calculated based on the code from this [source](https://github.com/Sundar0989/Variable-Selection-Using-Python/blob/master/Variable%20Selection%20using%20Python%20-%20Vote%20based%20approach.ipynb)

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

Local settings for model training in **train.py** in **fit** function:
1. **bayesian_optimisation** - True/False - True denotes that Bayesian Optimisation will be used to tune hyperparameters, with Precision-Recall AUC as a measure.
2. **random_search** - True/False - True denotes that Random Search will be used to tune hyperparameters, with Precision-Recall AUC as a measure.
3. **apply_smote** - True/False - If True, SMOTE algorithm will be used to transform train sample before a model is calculated. This is due to highly imbalanced dataset.

False for both **bayesian_optimisation** and **random_search** means that a model with hyperparameters saved in **train_ml_model.py** in **current_hyperparameters** will be used.

You can use only one method at the same time. It means, if **bayesian_optimisation** and **random_search** at the same time, only Bayesian Optimisation will be launched.

**Output**

As a result, xgb_model is serialized and saved in **data/trained_instances** folder. Moreover, files required for ExplainerDashboard for related model are saved in the project path.

If you want to run ExplainerDashboard for the model, in **terminal** in your repo, run the code:

explainerdashboard run dashboard.yaml

Besides, in the console, you can see measures printed for train, test and OOT samples:

1. ML measures: ROC AUC, Gini, Precision-Recall AUC, Accuracy, F1 Score, Balanced accuracy, Precision, Recall, as well as Confusion matrix.
2. CRM measures: Hit Rate calculated on entire sample, as well as Hit Rates on top 2.5%, 5%, 10% of customers with the highest probabilities from the model, respectively. For those subsamples, Lift is additionally calculated, which is estimated as HR_topx%sumsample/HT_entirepopulation. 

IMPORTANT NOTICE: in order to calculate measures in the step 1, the following cut-off is considered: the smallest probability in top 5% subsample of train population.

**Results**

You can see results for all calculated scenarios [here](https://docs.google.com/spreadsheets/d/1CioEZp9BVxXqVABmrKE7Za8GV-j0IQ3L/edit?usp=sharing&ouid=100478302082861511986&rtpof=true&sd=true)
