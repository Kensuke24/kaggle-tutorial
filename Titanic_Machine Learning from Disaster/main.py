import pandas as pd
import numpy as np


train = pd.read_csv("data/train.csv")   # train.csvの読み込み
test = pd.read_csv("data/test.csv")   # test.csvの読み込み
'''
print(train.shape)  # train.csvの行、列数を表示

print(test.describe())  # 基本統計量の確認。
'''


def kesson_table(df):   # dataframe（train.csvとtest.csv）の欠損データをisnull()で探してカラムごとに返す関数
    null_val = df.isnull().sum()    # カラムごとの欠損数
    percent = 100 * df.isnull().sum()/len(df)   #
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(
    columns = {0 : '欠損数', 1 : '%'})
    return kesson_table_ren_columns
    kesson_table(train)
    kesson_table(test)

print(kesson_table(train))