'''
https://www.codexa.net/kaggle-titanic-beginner/を参考に勉強させていただきました。
'''

import pandas as pd
import numpy as np
from sklearn import tree


train = pd.read_csv("data/train.csv")   # train.csvの読み込み
test = pd.read_csv("data/test.csv")   # test.csvの読み込み
'''
print(train.shape)  # train.csvの行、列数を表示

print(test.describe())  # 基本統計量の確認。
'''


def kesson_table(df):   # dataframe（train.csvとtest.csv）の欠損データをisnull()で探してカラムごとに返す関数
    null_val = df.isnull().sum()    # カラムごとの欠損数
    percent = 100 * df.isnull().sum()/len(df)   # 全体に対する欠損数の割合
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(columns = {0 : '欠損数', 1 : '%'})
    return kesson_table_ren_columns
    kesson_table(train)
    kesson_table(test)


train["Age"] = train["Age"].fillna(train["Age"].median())   # Ageカラムの欠損値にAgeの中央値を入れる
train["Embarked"] = train["Embarked"].fillna("S")   # Embarkedカラムの欠損値にSを入れる
kesson_table(train)

train["Sex"][train["Sex"] == "male"] = 0    # Sexカラムの文字列maleを0に置き換える
train["Sex"][train["Sex"] == "female"] = 1    # Sexカラムの文字列femaleを1に置き換える
train["Embarked"][train["Embarked"] == "S" ] = 0    # Embarkedカラムの文字列Sを0に置き換える
train["Embarked"][train["Embarked"] == "C" ] = 1    # Embarkedカラムの文字列Cを1に置き換える
train["Embarked"][train["Embarked"] == "Q"] = 2    # Embarkedカラムの文字列Qを2に置き換える

test["Age"] = test["Age"].fillna(test["Age"].median())
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2
test.Fare[152] = test.Fare.median()   # Fareカラムの欠損値(152)にFareの中央値を入れる


# 「train」の目的変数と説明変数の値を取得
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values
# 決定木の作成
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)
# 「test」の説明変数の値を取得
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values
# 「test」の説明変数を使って「my_tree_one」のモデルで予測
my_prediction = my_tree_one.predict(test_features)
