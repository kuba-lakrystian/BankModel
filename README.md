# BankModel

**Content**

The repo contains a package for development of the propensity model for credit cards based on dataset from Santander.

The dataset can be downloaded from [here](https://www.kaggle.com/competitions/santander-product-recommendation/data) - "train_ver2.csv.zip"

**Target definition**:

In a particular month (denoted as t) as a success ("1") we mark a customer who did not have a credit card three months before month t (it means, did not have a credit card in months t-3, t-2 and t-1), and have a credit card in a month t and one more month (t+1).
This is in order to avoid so-called "empty sale" - customers who bought a credit card, but did not use it in practice.

**How to launch the code?**

1. Create a new conda repository with the settings contained in **environment.yml**
2. Inside the project folder, create folder **data**. Inside folder data, create also folder **data_recommendation_engine**.
3. Download **train_ver2.csv.zip** from the link above and paste it in **data_recommendation_engine**. Unpack it here.
4. In **data** folder, create empty folder **trained_instances** - in this folder, serialized model will be saved.
5. Run **train.py** script and wait for the results

