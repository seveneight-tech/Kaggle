#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


get_ipython().system('pip install seaborn')


# In[3]:


get_ipython().system('pip install scikit-learn')


# In[ ]:


get_ipython().system('pip install lightgbm')


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")
submission_df = pd.read_csv("./data/gender_submission.csv")


# In[ ]:


print(test_df.shape)


# In[ ]:


test_df.dtypes


# In[ ]:


train_df["Cabin"].value_counts()


# In[ ]:


train_df.isnull().sum()


# In[ ]:


test_df.isnull().sum()


# In[ ]:


plt.style.use("ggplot")


# In[ ]:


embarked_df = train_df[["Embarked", "Survived", "PassengerId"]].dropna().groupby(["Embarked", "Survived"]).count().unstack()
embarked_df


# In[ ]:


embarked_df.plot.bar(stacked=True)


# In[ ]:


embarked_df["Survived_rate"] = embarked_df.iloc[:, 1] / (embarked_df.iloc[:, 0] + embarked_df.iloc[:, 1])


# In[ ]:


embarked_df


# In[ ]:


sex_df = train_df[["Sex", "Survived", "PassengerId"]].dropna().groupby(["Sex", "Survived"]).count().unstack()
ticket_df = train_df[["Pclass", "Survived", "PassengerId"]].dropna().groupby(["Pclass", "Survived"]).count().unstack()


# In[ ]:


sex_df["Survived_rate"] = sex_df.iloc[:, 1] / (sex_df.iloc[:, 0] + sex_df.iloc[:, 1])
ticket_df["Survived_rate"] = ticket_df.iloc[:, 1] / (ticket_df.iloc[:, 0] + ticket_df.iloc[:, 1])
sex_df


# In[ ]:


ticket_df


# In[ ]:


train_df_corr = pd.get_dummies(train_df, columns=["Embarked"])
train_df_corr = pd.get_dummies(train_df_corr, columns=["Sex"], drop_first=True)


# In[ ]:


train_df_corr = train_df_corr.corr()


# In[ ]:


train_df_corr


# In[ ]:


plt.figure(figsize=(9, 9))
sns.heatmap(train_df_corr, vmax=1, vmin=-1, center=0, annot=True)


# In[ ]:


all_df = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)
all_df


# In[ ]:


all_df.isnull().sum()


# In[ ]:


Fare_mean = all_df[["Pclass", "Fare"]].groupby("Pclass").mean().reset_index()
Fare_mean.columns = ["Pclass", "Fare_mean"]
Fare_mean


# In[ ]:


all_df = pd.merge(all_df, Fare_mean, on="Pclass", how="left")
all_df.loc[(all_df["Fare"].isnull()), "Fare"] = all_df["Fare_mean"]
all_df = all_df.drop("Fare_mean", axis=1)
all_df


# In[ ]:


name_df = all_df["Name"].str.split("[,.]", n=2, expand=True)
name_df.columns = ["family_name", "honorific", "name"]
name_df


# In[ ]:


name_df["family_name"] = name_df["family_name"].str.strip()
name_df["honorific"] = name_df["honorific"].str.strip()
name_df["name"] = name_df["name"].str.strip()


# In[ ]:


name_df["honorific"].value_counts()


# In[ ]:


all_df = pd.concat([all_df, name_df], axis=1)
all_df


# In[ ]:


plt.figure(figsize=(18, 5))
sns.boxplot(x = "honorific", y = "Age", data = all_df)


# In[ ]:


all_df[["Age", "honorific"]].groupby("honorific").mean()


# In[ ]:


train_name_df = pd.concat([train_df, name_df[:len(train_df)].reset_index(drop=True)], axis=1)
test_name_df = pd.concat([test_df, name_df[:len(test_df)].reset_index(drop=True)], axis=1)


# In[ ]:


honorific_df = train_name_df[["honorific", "Survived", "PassengerId"]].dropna().groupby(["honorific", "Survived"]).count().unstack()
honorific_df.plot.bar(stacked=True)


# In[ ]:


honorific_age_mean = all_df[["honorific", "Age"]].groupby("honorific").mean().reset_index()
honorific_age_mean.columns = ["honorific", "honorific_Age"]


# In[ ]:


all_df = pd.merge(all_df, honorific_age_mean, on="honorific", how="left")
all_df.loc[(all_df["Age"].isnull()), "Age"] = all_df["honorific_Age"]
all_df = all_df.drop(["honorific_Age"], axis=1)
all_df.isnull().sum()


# In[ ]:


all_df["family_num"] = all_df["Parch"] + all_df["SibSp"] + 1
all_df["family_num"].value_counts()


# In[ ]:


all_df.loc[all_df["family_num"] == 1, "alone"] = 1
all_df["alone"].fillna(0, inplace=True)


# In[ ]:


all_df


# In[ ]:


all_df = all_df.drop(["PassengerId", "Name", "family_name", "name", "Ticket", "Cabin"], axis=1)
all_df


# In[ ]:


categories = all_df.columns[all_df.dtypes == "object"]
print(categories)


# In[ ]:


all_df.loc[~((all_df["honorific"] == "Mr") | (all_df["honorific"] == "Miss") | (all_df["honorific"] == "Mrs") | (all_df["honorific"] == "Master")), "honorific"] = "other"


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


all_df["Embarked"].fillna("missing", inplace=True)


# In[ ]:


all_df.head()


# In[ ]:


for cat in categories:
    le = LabelEncoder()
    print(cat)
    if all_df[cat].dtypes == "object":
        le = le.fit(all_df[cat])
        all_df[cat] = le.transform(all_df[cat])


# In[ ]:


all_df


# In[ ]:


train_X = all_df[~all_df["Survived"].isnull()].drop("Survived", axis=1).reset_index(drop=True)
train_Y = train_df["Survived"]

test_X = all_df[all_df["Survived"].isnull()].drop("Survived", axis=1).reset_index(drop=True)


# In[ ]:


import lightgbm as lgb


# In[ ]:


#lightgbmはcondaコマンドからインストールした　homebrewだとosエラーが起こる


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_Y, test_size=0.2)


# In[ ]:


categories = ["Embarked", "Pclass", "Sex", "honorific", "alone"]


# In[ ]:


X_train[categories], X_valid[categories] = X_train[categories].astype("category"), X_valid[categories].astype("category")


# In[ ]:


def make_model(X_train, y_train, X_valid, y_valid):
    lgbm_params = {
        "objective" : "binary",
        "random_seed" : 1234,
        "force_col_wise" : True
    }
    
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
    
    model_lgb = lgb.train(
        lgbm_params,
        lgb_train,
        valid_sets=lgb_eval,
        num_boost_round=100,
        callbacks=[lgb.early_stopping(stopping_rounds=10, 
                            verbose=True), # early_stopping用コールバック関数
                            lgb.log_evaluation(10)] # 学習時のスコア推移が指定回数ごとにコマンドライン表示される
    )
    
    return model_lgb


# In[ ]:


model_lgb = make_model(X_train, y_train, X_valid, y_valid)


# In[ ]:


importance = pd.DataFrame(model_lgb.feature_importance(), index=X_train.columns, columns=["importance"]).sort_values(by="importance", ascending = True)
importance.plot.barh()


# In[ ]:


y_pred = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)


# In[ ]:


from sklearn.metrics import accuracy_score

accuracy_score(y_valid, np.round(y_pred))


# In[ ]:


# クロスバリデーションによる学習
from sklearn.model_selection import KFold

folds = 3
kf = KFold(n_splits=folds)

models = []
acc_score = []

for train_index, val_index in kf.split(train_X):
    X_train = train_X.iloc[train_index]
    Y_train = train_Y.iloc[train_index]
    X_valid = train_X.iloc[val_index]
    Y_valid = train_Y.iloc[val_index]
    
    model_lgb_KFold = make_model(X_train, Y_train,  X_valid, Y_valid)
    
    Y_pred = model_lgb_KFold.predict(X_valid, num_iteration=model_lgb_KFold.best_iteration)
    acc_score.append(accuracy_score(Y_valid, np.round(Y_pred)))
    models.append(model_lgb_KFold)

print(acc_score)


# In[ ]:


preds = []

for model in models:
    pred = model.predict(test_X)
    preds.append(pred)


# In[ ]:


preds_array = np.array(preds)
preds_mean = np.mean(preds_array, axis=0)
preds_int = (preds_mean > 0.5).astype(int)
preds_int


# In[ ]:


submission_df["Survived"] = preds_int
submission_df


# In[ ]:


submission_df.to_csv("./submit/titanic_submit01.csv", index=False)


# In[ ]:




