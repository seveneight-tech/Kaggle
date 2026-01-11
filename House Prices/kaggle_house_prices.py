#!/usr/bin/env python
# coding: utf-8

# #### 本Notebookは書籍『Pythonで動かして学ぶ！Kaggleデータ分析入門』(翔泳社, 2020)の内容のサンプルコードとなります。

# 本書には記載していないコードですが、ここから実行してください

# In[9]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:





# ## 4.3　データを取得する

# #### 必要なライブラリをインポートする

# リスト4.1 matplotlibとseabornのインポートとグラフ描画の設定

# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[11]:


plt.style.use("ggplot")


# リスト4.2　pandas、NumPyのインポート

# In[12]:


import pandas as pd
import numpy as np


# #### ランダムシードの設定

# リスト4.3　ランダムシードを設定

# In[13]:


import random
np.random.seed(1234)
random.seed(1234)


# #### CSVデータを読み込む（Anaconda（Windows）、 macOSでJupyter Notebookを利用する場合）

# リスト4.4　CSVデータの読み込み（Anaconda（Windows）、macOSでJupyter Notebookを利用する場合）

# In[14]:


train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")
submission = pd.read_csv("./data/sample_submission.csv")


# In[15]:


train_df.head()


# ## 4.4 ベースライン（ベンチマーク）を作成する

# ### LightGBMで予測する

# #### 学習データの各変数の型を確認する

# リスト4.6　各変数の型の確認

# In[16]:


train_df.dtypes


# リスト4.7　MSZoningの各分類ごとの個数を確認する

# In[17]:


train_df["MSZoning"].value_counts()


# ### 学習データとテストデータを連結して前処理を行う

# リスト4.8　学習データとテストデータの連結

# In[18]:


all_df = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)


# In[19]:


all_df


# リスト4.9　目的変数であるSalePriceの値を確認

# In[20]:


all_df["SalePrice"]


# #### カテゴリ変数を数値に変換する

# リスト4.10　LabelEncoderのライブラリをインポート

# In[21]:


from sklearn.preprocessing import LabelEncoder


# リスト4.11　object型の変数を取得

# In[22]:


categories = all_df.columns[all_df.dtypes == "object"]
print(categories)


# リスト4.12　'Alley'の各分類の個数を確認

# In[23]:


all_df["Alley"].value_counts()


# #### 欠損値を数値に変換する

# リスト4.13　欠損値を数値に変換

# In[24]:


for cat in categories:
    le = LabelEncoder()
    print(cat)

    all_df[cat].fillna("missing", inplace=True)
    le = le.fit(all_df[cat])
    all_df[cat] = le.transform(all_df[cat])
    all_df[cat] = all_df[cat].astype("category")


# In[25]:


all_df


# #### 再び学習データとテストデータに戻す

# リスト4.14　データをtrain_dfとtest_dfに戻す

# In[26]:


train_df_le = all_df[~all_df["SalePrice"].isnull()]
test_df_le = all_df[all_df["SalePrice"].isnull()]


# #### LightGBMに上記データを読み込ませる

# リスト4.15　LightGBMのライブラリをインポート

# In[27]:


import lightgbm as lgb


# ### クロスバリデーションを用いてモデルの学習・予測を行う

# #### クロスバリデーション用のライブラリを読み込んで分割数を設定する

# リスト4.16　クロスバリデーション用のライブラリを読み込んで分割数を3に設定

# In[28]:


from sklearn.model_selection import KFold
folds = 3
kf = KFold(n_splits=folds)


# #### ハイパーパラメータを設定する

# リスト4.17　LightGBMのハイパーパラメータを設定

# In[29]:


lgbm_params = {
    "objective":"regression",
    "random_seed":1234,
    "force_row_wise":True
}


# #### 説明変数、目的変数を指定する

# リスト4.18　説明変数、目的変数を指定

# In[30]:


train_X = train_df_le.drop(["SalePrice", "Id"], axis=1)
train_Y = train_df_le["SalePrice"]


# #### 平均二乗誤差を出すライブラリをインポートする

# リスト4.19　平均二乗誤差を出すライブラリをインポート

# In[31]:


from sklearn.metrics import mean_squared_error


# #### 各foldごとに作成したモデルごとの予測値を保存する

# リスト4.20　各foldごとに作成したモデルごとの予測値を保存

# In[32]:


models = []
rmses = []
oof = np.zeros(len(train_X))

for train_index, val_index in kf.split(train_X):
    X_train = train_X.iloc[train_index]
    X_valid = train_X.iloc[val_index]
    y_train = train_Y.iloc[train_index]
    y_valid = train_Y.iloc[val_index]

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    model_lgb = lgb.train(lgbm_params,
                          lgb_train,
                          valid_sets=lgb_eval,
                          num_boost_round=100,
                         )

    y_pred = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)
    tmp_rmse = np.sqrt(mean_squared_error(np.log(y_valid), np.log(y_pred)))
    print(tmp_rmse)

    models.append(model_lgb)
    rmses.append(tmp_rmse)
    oof[val_index] = y_pred


# #### 平均RMSEを計算する

# リスト4.21　平均RMSEを計算

# In[33]:


sum(rmses)/len(rmses)


# #### 現状の予測値と実際の値の違いを確認する

# リスト4.23　現状の予測値と実際の値の違いを可視化

# In[34]:


actual_pred_df = pd.DataFrame({
"actual" : train_Y,
"pred" : oof })


# In[35]:


actual_pred_df.plot(figsize=(12,5))


# ### 各変数の重要度を確認する

# #### 表示する変数の数を制限する

# リスト4.24　変数の数を制限して各変数の重要度を表示

# In[36]:


for model in models:
    lgb.plot_importance(model,importance_type="gain", max_num_features=15)


# ## 4.5　目的変数の前処理：目的変数の分布を確認する

# ### SalePriceのデータの分布を確認する

# #### SalePriceの各統計量を確認する

# リスト4.25　SalePriceの各統計量を確認

# In[37]:


train_df["SalePrice"].describe()


# #### ヒストグラムでSalePriceの分布を確認する

# リスト4.26　ヒストグラムで分布を確認

# In[38]:


train_df["SalePrice"].plot.hist(bins=20)


# #### 目的変数を対数化する

# リスト4.27　SalePriceを対数化

# In[39]:


np.log(train_df['SalePrice'])


# リスト4.28　対数化したSalePriceの分布をヒストグラムで可視化

# In[40]:


np.log(train_df['SalePrice']).plot.hist(bins=20)


# #### 目的変数の対数化による予測精度の向上を確認する

# リスト4.29　対数化による予測精度の向上を確認

# In[41]:


train_df_le["SalePrice_log"] = np.log(train_df_le["SalePrice"])


# In[42]:


train_X = train_df_le.drop(["SalePrice","SalePrice_log","Id"], axis=1)
train_Y = train_df_le["SalePrice_log"]


# In[43]:


models = []
rmses = []
oof = np.zeros(len(train_X))

for train_index, val_index in kf.split(train_X):
    X_train = train_X.iloc[train_index]
    X_valid = train_X.iloc[val_index]
    y_train = train_Y.iloc[train_index]
    y_valid = train_Y.iloc[val_index]

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    model_lgb = lgb.train(lgbm_params,
                          lgb_train,
                          valid_sets=lgb_eval,
                          num_boost_round=100,
                         )

    y_pred = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)
    tmp_rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    print(tmp_rmse)

    models.append(model_lgb)
    rmses.append(tmp_rmse)
    oof[val_index] = y_pred


# In[44]:


sum(rmses)/len(rmses)


# ## 4.6　説明変数の前処理：欠損値を確認する

# ### 各説明変数の欠損値を確認する

# #### all_dfを作成する

# リスト4.30　all_dfの作成

# In[45]:


all_df = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)


# In[46]:


categories = all_df.columns[all_df.dtypes == "object"]
print(categories)


# #### 欠損値の数が上位40の変数を確認する

# リスト4.31　欠損値の数が上位40の変数を確認

# In[47]:


all_df.isnull().sum().sort_values(ascending=False).head(40)


# #### 欠損値の多い高級住宅設備に関する変数をまとめる

# リスト4.32　PoolQCの各分類ごとの個数

# In[48]:


all_df.PoolQC.value_counts()


# リスト4.33　PoolQCの値を値があるものを1、値がないものを0に変換

# In[49]:


all_df.loc[~all_df["PoolQC"].isnull(), "PoolQC"] = 1
all_df.loc[all_df["PoolQC"].isnull(), "PoolQC"] = 0


# リスト4.34　0か1の値を持つ項目になったかを確認

# In[50]:


all_df.PoolQC.value_counts()


# リスト4.35　MiscFeature、Alleyも0と1に変換する

# In[51]:


all_df.loc[~all_df["MiscFeature"].isnull(), "MiscFeature"] = 1
all_df.loc[all_df["MiscFeature"].isnull(), "MiscFeature"] = 0


# In[52]:


all_df.loc[~all_df["Alley"].isnull(), "Alley"] = 1
all_df.loc[all_df["Alley"].isnull(), "Alley"] = 0


# リスト4.36　繰り返し処理はfor文でまとめる

# In[53]:


HighFacility_col = ["PoolQC","MiscFeature","Alley"]
for col in HighFacility_col:
    if all_df[col].dtype == "object":
        if len(all_df[all_df[col].isnull()]) > 0:
            all_df.loc[~all_df[col].isnull(), col] = 1
            all_df.loc[all_df[col].isnull(), col] = 0


# リスト4.37　0か1の値に変換した各変数を足し合わせて、高級住宅設備の数という特徴量を作成

# In[54]:


all_df["hasHighFacility"] = all_df["PoolQC"] + all_df["MiscFeature"] + all_df["Alley"]


# In[55]:


all_df["hasHighFacility"] = all_df["hasHighFacility"].astype(int)


# リスト4.38　高級住宅設備の数ごとの家の数を確認

# In[56]:


all_df["hasHighFacility"].value_counts()


# リスト4.39　もとのデータからPoolQC、MiscFeature、Alleyを削除

# In[57]:


all_df = all_df.drop(["PoolQC","MiscFeature","Alley"],axis=1)


# ## 4.7 外れ値を除外する

# ### 各説明変数のデータの分布を確認する

# #### 各変数の統計量を確認する

# リスト4.40　各変数の統計量を確認

# In[58]:


all_df.describe().T


# #### 数値データのみを抜き出す

# リスト4.41　数値データのみの抜き出し

# In[59]:


train_df_num = train_df.select_dtypes(include=[np.number])


# リスト4.42　比例尺度ではない変数

# In[60]:


nonratio_features = ["Id", "MSSubClass", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "MoSold", "YrSold"]


# リスト4.43　数値データからリスト4.43の変数を除いた比例尺度データ

# In[61]:


num_features = sorted(list(set(train_df_num) - set(nonratio_features)))


# In[62]:


num_features


# リスト4.44　比例尺度の列のみを抜き出す

# In[63]:


train_df_num_rs = train_df_num[num_features]


# #### 多数のデータが0（ゼロ）の値である変数を確認する

# リスト4.45　3/4分位数が0となる変数を確認

# In[64]:


for col in num_features:
    if train_df_num_rs.describe()[col]["75%"] == 0:
        print(col, len(train_df_num_rs[train_df_num_rs[col] == 0]))


# #### ある特定の値のみを持つ変数を確認する

# リスト4.46　ある特定の値のみしかとらないものを確認

# In[65]:


for col in num_features:
    if train_df_num_rs[col].nunique() < 15:
        print(col, train_df_num_rs[col].nunique())


# #### 外れ値があるか確認する

# リスト4.47　外れ値があるか確認

# In[66]:


for col in num_features:
    tmp_df = train_df_num_rs[(train_df_num_rs[col] > train_df_num_rs[col].mean() + train_df_num_rs[col].std()*3) | \
    (train_df_num_rs[col] < train_df_num_rs[col].mean() - train_df_num_rs[col].std()*3)]
    print(col, len(tmp_df))


# #### 外れ値を含む変数の分布を可視化する

# リスト4.48　BsmtFinSF1とSalePriceの分布を可視化

# In[67]:


all_df.plot.scatter(x="BsmtFinSF1", y="SalePrice")


# リスト4.49　BsmtFinSF1が広いもののSalePriceが高くないものを確認

# In[68]:


all_df[all_df["BsmtFinSF1"] > 5000]


# リスト4.50　TotalBsmtSFとSalePriceの分布を可視化

# In[69]:


all_df.plot.scatter(x="TotalBsmtSF", y="SalePrice")


# In[70]:


all_df[all_df["TotalBsmtSF"] > 6000]


# リスト4.51　GrLivAreaとSalePriceの分布を可視化

# In[71]:


all_df.plot.scatter(x="GrLivArea", y="SalePrice")


# In[72]:


all_df[all_df["GrLivArea"] > 5000]


# リスト4.52　1stFlrSFとSalePriceの分布を可視化

# In[73]:


all_df.plot.scatter(x="1stFlrSF", y="SalePrice")


# In[74]:


all_df[all_df["1stFlrSF"] > 4000]


# リスト4.53　外れ値以外を抽出（テストデータはすべて抽出）

# In[75]:


all_df = all_df[(all_df['BsmtFinSF1'] < 2000) | (all_df['SalePrice'].isnull())]
all_df = all_df[(all_df['TotalBsmtSF'] < 3000) | (all_df['SalePrice'].isnull())]
all_df = all_df[(all_df['GrLivArea'] < 4500) | (all_df['SalePrice'].isnull())]
all_df = all_df[(all_df['1stFlrSF'] < 2500) | (all_df['SalePrice'].isnull())]
all_df = all_df[(all_df['LotArea'] < 100000) | (all_df['SalePrice'].isnull())]


# #### 前処理した学習データでRMSEを計算する

# リスト4.54　categoriesの中から除外した3つの変数を削除

# In[76]:


categories = categories.drop(["PoolQC","MiscFeature","Alley"])


# リスト4.55　欠損値をmissingに置き換えてall_dfのカテゴリ変数をcategoryに指定

# In[77]:


for cat in categories:
    le = LabelEncoder()
    print(cat)

    all_df[cat].fillna("missing", inplace=True)
    le = le.fit(all_df[cat])
    all_df[cat] = le.transform(all_df[cat])
    all_df[cat] = all_df[cat].astype("category")


# リスト4.56　train_df_leとtest_df_leに分割

# In[78]:


train_df_le = all_df[~all_df["SalePrice"].isnull()]
test_df_le = all_df[all_df["SalePrice"].isnull()]

train_df_le["SalePrice_log"] = np.log(train_df_le["SalePrice"])
train_X = train_df_le.drop(["SalePrice","SalePrice_log", "Id"], axis=1)
train_Y = train_df_le["SalePrice_log"]


# In[79]:


models = []
rmses = []
oof = np.zeros(len(train_X))

for train_index, val_index in kf.split(train_X):
    X_train = train_X.iloc[train_index]
    X_valid = train_X.iloc[val_index]
    y_train = train_Y.iloc[train_index]
    y_valid = train_Y.iloc[val_index]

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    model_lgb = lgb.train(lgbm_params,
                          lgb_train,
                          valid_sets=[lgb_eval],
                          num_boost_round=100,
                         )

    y_pred = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)
    tmp_rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    print(tmp_rmse)

    models.append(model_lgb)
    rmses.append(tmp_rmse)
    oof[val_index] = y_pred


# In[80]:


print(sum(rmses)/len(rmses))


# ## 4.8 　説明変数の確認：特徴量を生成する

# #### 時間に関する変数の統計量を確認する

# リスト4.57　時間に関する変数の統計量を確認

# In[81]:


all_df = all_df[~(all_df["GarageYrBlt"] > 2025)]


# In[82]:


all_df[["YearBuilt","YearRemodAdd","GarageYrBlt","YrSold"]].describe()


# #### 時間に関する変数を組み合わせて新たな特徴量を作成する

# リスト4.58　特徴量を追加

# In[83]:


all_df["Age"] = all_df["YrSold"] - all_df["YearBuilt"]


# In[84]:


train_df_le = all_df[~all_df["SalePrice"].isnull()]
test_df_le = all_df[all_df["SalePrice"].isnull()]

train_df_le["SalePrice_log"] = np.log(train_df_le["SalePrice"])
train_X = train_df_le.drop(["SalePrice","SalePrice_log","Id"], axis=1)
train_Y = train_df_le["SalePrice_log"]


# In[85]:


models = []
rmses = []
oof = np.zeros(len(train_X))

for train_index, val_index in kf.split(train_X):
    X_train = train_X.iloc[train_index]
    X_valid = train_X.iloc[val_index]
    y_train = train_Y.iloc[train_index]
    y_valid = train_Y.iloc[val_index]

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    model_lgb = lgb.train(lgbm_params,
                          lgb_train,
                          valid_sets=[lgb_eval],
                          num_boost_round=100,
                         )

    y_pred = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)
    tmp_rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    print(tmp_rmse)

    models.append(model_lgb)
    rmses.append(tmp_rmse)
    oof[val_index] = y_pred


# In[86]:


sum(rmses)/len(rmses)


# リスト4.59　他の変数を追加

# In[87]:


# 販売した年はリノベーションしてから何年経過していたか（リノベーションしていない場合、Age（築年数）と同じ）
#all_df["RmdAge"] = all_df["YrSold"] - all_df["YearRemodAdd"]
# 販売した年はガレージ建築から何年経過していたか
#all_df["GarageAge"] = all_df["YrSold"] - all_df["GarageYrBlt"]
# 築何年たってから、リノベーションしたか
#all_df["RmdTiming"] = all_df["YearRemodAdd"] - all_df["YearBuilt"]


# #### 広さ関連の変数から新たな特徴量を作成する

# リスト4.60　広さに関する変数の統計量を確認

# In[88]:


all_df[["LotArea","MasVnrArea","BsmtUnfSF","TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "GarageArea","WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "LotFrontage"]].describe()


# リスト4.61　広さの変数から追加するもの

# In[89]:


all_df["TotalSF"] = all_df["TotalBsmtSF"] + all_df["1stFlrSF"] + all_df["2ndFlrSF"]
all_df["Total_Bathrooms"] = all_df["FullBath"] + all_df["HalfBath"] + all_df["BsmtFullBath"] + all_df["BsmtHalfBath"]


# リスト4.62　Porchの広さの合計も特徴量として追加

# In[90]:


all_df["Total_PorchSF"] = all_df["WoodDeckSF"] + all_df["OpenPorchSF"] + all_df["EnclosedPorch"] + all_df["3SsnPorch"] + all_df["ScreenPorch"]


# リスト4.63　Porchの広さの合計をPorchがあるかないかの0、1の値に変換

# In[91]:


all_df["hasPorch"] = all_df["Total_PorchSF"].apply(lambda x: 1 if x > 0 else 0)
all_df = all_df.drop("Total_PorchSF",axis=1)


# リスト4.64　精度を確認

# In[92]:


train_df_le = all_df[~all_df["SalePrice"].isnull()]
test_df_le = all_df[all_df["SalePrice"].isnull()]

train_df_le["SalePrice_log"] = np.log(train_df_le["SalePrice"])
train_X = train_df_le.drop(["SalePrice","SalePrice_log","Id"], axis=1)
train_Y = train_df_le["SalePrice_log"]


# In[93]:


models = []
rmses = []
oof = np.zeros(len(train_X))

for train_index, val_index in kf.split(train_X):
    X_train = train_X.iloc[train_index]
    X_valid = train_X.iloc[val_index]
    y_train = train_Y.iloc[train_index]
    y_valid = train_Y.iloc[val_index]

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    model_lgb = lgb.train(lgbm_params,
                          lgb_train,
                          valid_sets=[lgb_eval],
                          num_boost_round=100,
                         )

    y_pred = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)
    tmp_rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    print(tmp_rmse)

    models.append(model_lgb)
    rmses.append(tmp_rmse)
    oof[val_index] = y_pred


# In[94]:


sum(rmses)/len(rmses)


# ## 4.9　ハイパーパラメータを最適化する 6/9 ここから

# #### Optunaのライブラリをインストール・インポートする

# リスト4.65　Optunaのライブラリのインポート

# In[95]:


get_ipython().system('pip3 install optuna')


# In[96]:


import optuna


# ### Optunaを実装する

# #### 学習データ、検証データを作成する

# リスト4.66　学習データと検証データを作成

# In[97]:


from sklearn.model_selection import train_test_split


# In[98]:


X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_Y, test_size=0.2, random_state=1234, shuffle=False,  stratify=None)


# #### ハイパーパラメータを最適化する

# リスト4.67　Optunaでハイパーパラメータを最適化する

# In[99]:


def objective(trial):
    params = {
        "objective":"regression",
        "random_seed":1234,
        "learning_rate":0.05,
        "n_estimators":1000,
        "force_col_wise":True,

        "num_leaves":trial.suggest_int("num_leaves",4,64),
        "max_bin":trial.suggest_int("max_bin",50,200),
        "bagging_fraction":trial.suggest_uniform("bagging_fraction",0.4,0.9),
        "bagging_freq":trial.suggest_int("bagging_freq",1,10),
        "feature_fraction":trial.suggest_uniform("feature_fraction",0.4,0.9),
        "min_data_in_leaf":trial.suggest_int("min_data_in_leaf",2,16),
        "min_sum_hessian_in_leaf":trial.suggest_int("min_sum_hessian_in_leaf",1,10),
    }

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    model_lgb = lgb.train(params, lgb_train,
                          valid_sets=lgb_eval,
                          num_boost_round=100,)

    y_pred = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)
    score =  np.sqrt(mean_squared_error(y_valid, y_pred))

    return score


# In[100]:


study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
study.optimize(objective, n_trials=50)
study.best_params


# リスト4.68　得られたハイパーパラメータを設定してクロスバリデーション

# In[ ]:


lgbm_params = {
    "objective":"regression",
    "random_seed":1234,
    "force_col_wise":True,
    "learning_rate":0.05,
    "n_estimators":1000,
    "num_leaves":12,
    "bagging_fraction": 0.8319278029616157,
    "bagging_freq": 5,
    "feature_fraction": 0.4874544371547538,
    "max_bin":189,
    "min_data_in_leaf":13,
    "min_sum_hessian_in_leaf":4
}


# In[ ]:


models = []
rmses = []
oof = np.zeros(len(train_X))

for train_index, val_index in kf.split(train_X):
    X_train = train_X.iloc[train_index]
    X_valid = train_X.iloc[val_index]
    y_train = train_Y.iloc[train_index]
    y_valid = train_Y.iloc[val_index]

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    model_lgb = lgb.train(lgbm_params,
                          lgb_train,
                          valid_sets=[lgb_eval],
                          num_boost_round=100,
                         )

    y_pred = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)
    tmp_rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    print(tmp_rmse)

    models.append(model_lgb)
    rmses.append(tmp_rmse)
    oof[val_index] = y_pred


# In[ ]:


sum(rmses)/len(rmses)


# ### Kaggleに結果をsubmitする

# #### テストデータを用意する

# リスト4.69　テストデータを用意

# In[ ]:


test_X = test_df_le.drop(["SalePrice", "Id"], axis=1)


# #### 学習したモデルでテストデータの目的変数を予測する

# リスト4.70　クロスバリデーションごとの各モデルで予測値を算出

# In[ ]:


preds = []

for model in models:
    pred = model.predict(test_X)
    preds.append(pred)


# リスト4.71　predsの平均を計算してpreds_meanとして取得

# In[ ]:


preds_array = np.array(preds)
preds_mean = np.mean(preds_array, axis=0)


# #### 予測値をもとのスケールに戻す

# リスト4.72　もとのスケールに戻す

# In[ ]:


preds_exp = np.exp(preds_mean)


# In[ ]:


len(preds_exp)


# #### 予測値からsubmissionファイルを作成する

# リスト4.73　予測値をSalePriceの値として置き換え

# In[ ]:


submission["SalePrice"] = preds_exp


# #### CSVファイルとして書き出す

# リスト4.74　CSVファイルとして書き出す（Anaconda（Windows）、macOSでJupyterNotebookを利用する場合）

# In[ ]:


submission.to_csv("./submit/houseprices_submit01.csv",index=False)


# ## 4.10 様々な機械学習手法によるアンサンブル

# ### ランダムフォレストで学習する

# #### ランダムフォレストのライブラリを読み込む

# リスト4.76　ランダムフォレスト用のライブラリの読み込み

# In[ ]:


from sklearn.ensemble import RandomForestRegressor as rf


# ### LotFrontageの欠損値を削除する

# #### 欠損値を含む変数を確認する

# リスト4.77　欠損値を含む変数を確認

# In[ ]:


hasnan_cat = []
for col in all_df.columns:
    tmp_null_count = all_df[col].isnull().sum()
    if (tmp_null_count > 0) & (col != "SalePrice"):
        print(col, tmp_null_count)
        hasnan_cat.append(col)


# #### 欠損値を含む変数の統計量を確認する

# リスト4.78　hasnan_catに含まれる変数を確認

# In[ ]:


all_df[hasnan_cat].describe()


# #### 欠損値を各変数の中央値で補完する

# リスト4.79　欠損値を各変数の中央値で補完

# In[ ]:


for col in all_df.columns:
    tmp_null_count = all_df[col].isnull().sum()
    if (tmp_null_count > 0) & (col != "SalePrice"):
        print(col, tmp_null_count)
        all_df[col] = all_df[col].fillna(all_df[col].median())


# #### ランダムフォレストを用いて学習・予測する

# リスト4.80　SalePriceの対数をとって学習

# In[ ]:


train_df_le = all_df[~all_df["SalePrice"].isnull()]
test_df_le = all_df[all_df["SalePrice"].isnull()]
train_df_le["SalePrice_log"] = np.log(train_df_le["SalePrice"])


# In[ ]:


train_X = train_df_le.drop(["SalePrice","SalePrice_log","Id"], axis=1)
train_Y = train_df_le["SalePrice_log"]


# In[ ]:


folds = 3
kf = KFold(n_splits=folds)


# In[ ]:


models_rf = []
rmses_rf = []
oof_rf = np.zeros(len(train_X))
for train_index, val_index in kf.split(train_X):
    X_train = train_X.iloc[train_index]
    X_valid = train_X.iloc[val_index]
    y_train = train_Y.iloc[train_index]
    y_valid = train_Y.iloc[val_index]
    model_rf = rf(
        n_estimators=50,
        random_state=1234
    )
    model_rf.fit(X_train, y_train)
    y_pred = model_rf.predict(X_valid)
    tmp_rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    print(tmp_rmse)
    models_rf.append(model_rf)
    rmses_rf.append(tmp_rmse)
    oof_rf[val_index] = y_pred


# In[ ]:


sum(rmses_rf)/len(rmses_rf)


# #### 結果をCSVファイルとして書き出す

# リスト4.81　テストデータで各クロスバリデーションのモデルで予測値を算出

# In[ ]:


test_X = test_df_le.drop(["SalePrice","Id"], axis=1)


# In[ ]:


preds_rf = []
for model in models_rf:
    pred = model.predict(test_X)
    preds_rf.append(pred)


# In[ ]:


preds_array_rf = np.array(preds_rf)
preds_mean_rf = np.mean(preds_array_rf, axis=0)
preds_exp_rf = np.exp(preds_mean_rf)
submission["SalePrice"] = preds_exp_rf


# リスト4.82　CSVファイルの書き出し（Anaconda（Windows）、macOSでJupyterNotebookを利用する場合）

# In[ ]:


submission.to_csv("./submit/houseprices_submit02.csv",index=False)


# ### XGBoostで学習する

# #### XGBoostのライブラリをインストール・インポートする

# リスト4.84　XGBoostのライブラリのインポート

# In[ ]:


import xgboost as xgb


# #### XGBoostを実装する

# リスト4.85　category変数をint型に変換する

# In[ ]:


categories = train_X.columns[train_X.dtypes == "category"]


# In[ ]:


for col in categories:
    train_X[col] = train_X[col].astype("int8")
    test_X[col] = test_X[col].astype("int8")


# #### Optunaでハイパーパラメータを調整する

# リスト4.86　Optunaでハイパーパラメータを調整

# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_Y, test_size=0.2, random_state=1234, shuffle=False,  stratify=None)


# In[ ]:


def objective(trial):
    xgb_params = {
    "learning_rate":0.05,
    "seed":1234,
    "max_depth":trial.suggest_int("max_depth",3,16),
    "colsample_bytree":trial.suggest_uniform("colsample_bytree",0.2,0.9),
    "sublsample":trial.suggest_uniform("sublsample",0.2,0.9),
    }
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_eval = xgb.DMatrix(X_valid, label=y_valid)
    evals = [(xgb_train, "train"), (xgb_eval, "eval")]
    model_xgb = xgb.train(xgb_params, xgb_train,
    evals=evals,
    num_boost_round=1000,
    early_stopping_rounds=20,
    verbose_eval=10,)
    y_pred = model_xgb.predict(xgb_eval)
    score = np.sqrt(mean_squared_error(y_valid, y_pred))
    return score


# In[ ]:


study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
study.optimize(objective, n_trials=50)
study.best_params


# リスト4.87　ハイパーパラメータの設定

# In[ ]:


xgb_params = {
"learning_rate":0.05,
"seed":1234,
"max_depth": 6,
"colsample_bytree": 0.330432640328732,
"sublsample": 0.7158427239902707
}


# #### XGBoostでモデルを学習する

# リスト4.88　最適化の処理

# In[ ]:


models_xgb = []
rmses_xgb = []
oof_xgb = np.zeros(len(train_X))
for train_index, val_index in kf.split(train_X):
    X_train = train_X.iloc[train_index]
    X_valid = train_X.iloc[val_index]
    y_train = train_Y.iloc[train_index]
    y_valid = train_Y.iloc[val_index]
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_eval = xgb.DMatrix(X_valid, label=y_valid)
    evals = [(xgb_train, "train"), (xgb_eval, "eval")]
    model_xgb = xgb.train(xgb_params, xgb_train,
    evals=evals,
    num_boost_round=1000,
    early_stopping_rounds=20,
    verbose_eval=20,)
    y_pred = model_xgb.predict(xgb_eval)
    tmp_rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    print(tmp_rmse)
    models_xgb.append(model_xgb)
    rmses_xgb.append(tmp_rmse)
    oof_xgb[val_index] = y_pred


# In[ ]:


sum(rmses_xgb)/len(rmses_xgb)


# #### 結果をCSVファイルとして書き出す

# リスト4.89　テストデータでの予測値を算出

# In[ ]:


xgb_test = xgb.DMatrix(test_X)


# In[ ]:


preds_xgb = []
for model in models_xgb:
    pred = model.predict(xgb_test)
    preds_xgb.append(pred)


# In[ ]:


preds_array_xgb= np.array(preds_xgb)
preds_mean_xgb = np.mean(preds_array_xgb, axis=0)
preds_exp_xgb = np.exp(preds_mean_xgb)
submission["SalePrice"] = preds_exp_xgb


# リスト4.90　CSVファイルの書き出し（Anaconda（Windows）、macOSでJupyterNotebookを利用する場合）

# In[ ]:


submission.to_csv("./submit/houseprices_submit03.csv",index=False)


# ### XGBoostとLightGBMの結果を組み合わせる

# #### XGBoostの予測結果とLightGBMの予測結果の平均をとる

# リスト4.92　XgBoostの予測結果とLightGBMの予測結果の平均をとる

# In[ ]:


preds_ans = preds_exp_xgb * 0.5 + preds_exp * 0.5


# In[ ]:


submission["SalePrice"] = preds_ans


# リスト4.93　予測結果をCSVファイルとして書き出す（Anaconda（Windows）、macOSでJupyterNotebookを利用する場合）

# In[ ]:


submission.to_csv("./submit/houseprices_submit04.csv",index=False)


# ## 4.11　追加分析①統計手法による家のクラスタ分析を行う

# ### 統計手法を用いて家を分類する

# #### 欠損値のある行を削除する

# リスト4.95　欠損値がある行を削除する

# In[ ]:


train_df_le_dn = train_df_le.dropna()


# In[ ]:


train_df_le_dn


# #### データの正規化を行う

# リスト4.96　データの正規化

# In[ ]:


from sklearn import preprocessing


# In[ ]:


train_scaled = preprocessing.scale(train_df_le_dn.drop(["Id"],axis=1))


# In[ ]:


train_scaled


# #### np.array形式をDataFrame形式に戻す

# リスト4.97　np.array形式をDataFrame形式に戻す処理

# In[ ]:


train_scaled_df = pd.DataFrame(train_scaled)
train_scaled_df.columns = train_df_le_dn.drop(["Id"],axis=1).columns


# In[ ]:


train_scaled_df


# #### k-meansによるクラスタ分析

# #### k-means用のライブラリをインポートする

# リスト4.98　k-means用のライブラリのインポート

# In[ ]:


from sklearn.cluster import KMeans


# リスト4.99　ランダムシードを設定

# In[ ]:


np.random.seed(1234)


# リスト4.100　クラスタ数を指定して分類

# In[ ]:


house_cluster = KMeans(n_clusters=4).fit_predict(train_scaled)


# #### もとのデータにクラスタ情報を付与する

# リスト4.101　家ごとのクラスタ情報を追加

# In[ ]:


train_scaled_df["km_cluster"] = house_cluster


# #### クラスタごとのデータ数を確認

# リスト4.102　クラスタごとのデータ数を確認

# In[ ]:


train_scaled_df["km_cluster"].value_counts()


# #### クラスタごとの特徴を可視化する

# リスト4.103　クラスタごとの特徴を可視化

# In[ ]:


cluster_mean = train_scaled_df[["km_cluster","SalePrice","TotalSF","OverallQual","Age","Total_Bathrooms","YearRemodAdd","GarageArea",
                                "MSZoning","OverallCond","KitchenQual","FireplaceQu"]].groupby("km_cluster").mean().reset_index()


# リスト4.104　転置処理を施して可視化

# In[ ]:


cluster_mean = cluster_mean.T


# In[ ]:


cluster_mean


# In[ ]:


cluster_mean[1:].plot(figsize=(12,10), kind="barh" , subplots=True, layout=(1, 4) , sharey=True)


# ### 主成分分析を行う

# #### 主成分分析用のライブラリをインポートする

# リスト4.105　PCAパッケージのインポート

# In[ ]:


from sklearn.decomposition import PCA


# #### 標準化したデータに対して主成分分析を行う

# リスト4.106　主成分の数を指定

# In[ ]:


pca = PCA(n_components=2)
house_pca = pca.fit(train_scaled).transform(train_scaled)


# In[ ]:


house_pca


# #### 出力結果をDataFrame形式に変換してもとのDataFrameと結合する

# リスト4.107　出力結果をDataFrame形式に変換してもとのDataFrameと結合

# In[ ]:


house_pca_df = pd.DataFrame(house_pca)
house_pca_df.columns = ["pca1","pca2"]


# In[ ]:


train_scaled_df = pd.concat([train_scaled_df, house_pca_df], axis=1)


# In[ ]:


train_scaled_df


# #### 主成分分析の結果を可視化する

# リスト4.108　主成分分析の結果を可視化

# In[ ]:


my_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# In[ ]:


for cl in train_scaled_df['km_cluster'].unique():
    plt.scatter(train_scaled_df.loc[train_scaled_df["km_cluster"] == cl ,'pca1'], train_scaled_df.loc[train_scaled_df["km_cluster"] == cl ,'pca2'], label=cl, c=my_colors[cl], alpha=0.6)
plt.legend()
plt.show()


# リスト4.109　見やすいように転置（行と列を変換）

# In[ ]:


pca_comp_df = pd.DataFrame(pca.components_,columns=train_scaled_df.drop(["km_cluster","pca1","pca2"],axis=1).columns).T
pca_comp_df.columns = ["pca1","pca2"]


# In[ ]:


pca_comp_df


# ## 4.12　追加分析②ハイクラスな家の条件を分析・可視化する

# ### 決定木で可視化する

# #### SalePriceの分布を確認する

# リスト4.110　SalePriceの分布を確認

# In[ ]:


train_df_le['SalePrice'].plot.hist(bins=20)


# In[ ]:


train_df_le['SalePrice'].describe()


# #### 上位10%の価格を調べる

# リスト4.111　上位10%の価格を確認

# In[ ]:


train_df['SalePrice'].quantile(0.9)


# #### ハイクラスな家を表す変数を追加する

# リスト4.112　high_class変数を追加

# In[ ]:


train_df_le.loc[train_df["SalePrice"] >= 278000, "high_class"] = 1


# リスト4.113　条件を満たさないものを0とする

# In[ ]:


train_df_le["high_class"] = train_df_le["high_class"].fillna(0)


# In[ ]:


train_df_le.head()


# #### ライブラリをインポートする

# リスト4.115　ライブラリのインポート

# In[ ]:


from sklearn import tree
import pydotplus
from six import StringIO


# #### 重要度の高い変数に絞る

# リスト4.116　tree_xとtree_yを指定

# In[ ]:


tree_x = train_df_le[["TotalSF","OverallQual","Age","GrLivArea","GarageCars","Total_Bathrooms","GarageType",
"YearRemodAdd","GarageArea","CentralAir","MSZoning","OverallCond","KitchenQual","FireplaceQu","1stFlrSF"]]
tree_y = train_df_le[["high_class"]]


# #### 深さを指定して決定木を作成する

# リスト4.117　決定木の作成

# In[ ]:


clf = tree.DecisionTreeClassifier(max_depth=4)
clf = clf.fit(tree_x, tree_y)


# #### 決定木の出力結果を確認する

# リスト4.118　決定木の出力結果を確認

# In[ ]:


dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,feature_names=tree_x.columns)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())


# In[ ]:


from IPython.display import Image
Image(graph.create_png())

