import pandas as pd
import numpy as np

# 目的：處理所有資料


# 導入訓練集與測試集
train_df = pd.read_csv("Data/train.csv")
test_df = pd.read_csv("Data/test.csv")
combine = [train_df, test_df]

# ----------------------------------------------------------------

# 資料清理
# 刪除沒用的特徵
train_df = train_df.drop(["Ticket", "Cabin", "PassengerId"], axis=1)
test_df = test_df.drop(["Ticket", "Cabin", "PassengerId"], axis=1)
combine = [train_df, test_df]

# 創建新特徵（將 Name 的稱謂提取出來 Ex: Braund, Mr. Owen Harris 會提取 Mr）
for database in combine:
    database["Title"] = database.Name.str.extract(" ([a-zA-Z]+)\.", expand=False)

# 比較 Title 跟 Sex 的統計關係
compare = pd.crosstab(train_df["Title"], train_df["Sex"])
# print(compare)

# 將 Title 分成五種
for database in combine:
    database["Title"] = database["Title"].replace(
        [
            "Capt",
            "Col",
            "Countess",
            "Don",
            "Dr",
            "Jonkheer",
            "Lady",
            "Major",
            "Rev",
            "Sir",
        ],
        "Rare",
    )
    database["Title"] = database["Title"].replace("Mlle", "Miss")
    database["Title"] = database["Title"].replace("Ms", "Miss")
    database["Title"] = database["Title"].replace("Mme", "Miss")

# Title 與 Survived 關係
# 女性存活率較高
result = (
    train_df[["Title", "Survived"]]
    .groupby(["Title"], as_index=False)
    .mean()
    .sort_values(by="Survived", ascending=False)
    .to_string(index=False)
)
# print(result)

# 將 Name 刪除
train_df = train_df.drop(["Name"], axis=1)
test_df = test_df.drop(["Name"], axis=1)
combine = [train_df, test_df]

# ----------------------------------------------------------------

# 資料填充
# Embarked 跟 Age 有缺失值
# print(train_df.info())

# Embarked：填充最常出現的數值
freq = train_df.Embarked.dropna().mode()[0]
for database in combine:
    database.Embarked = database.Embarked.fillna(freq)

# Age：用 Pclass、Sex、Title 來預測
predictAge = train_df.groupby(["Pclass", "Sex", "Title"])["Age"].mean().reset_index()
# print(predictAge)


def fill_age(x):
    return predictAge[
        (predictAge["Pclass"] == x["Pclass"])
        & (predictAge["Sex"] == x["Sex"])
        & (predictAge["Title"] == x["Title"])
    ].Age.values[0]


train_df["Age"] = train_df.apply(
    lambda x: fill_age(x) if np.isnan(x["Age"]) else x["Age"], axis=1
)
test_df["Age"] = test_df.apply(
    lambda x: fill_age(x) if np.isnan(x["Age"]) else x["Age"], axis=1
)
combine = [train_df, test_df]

# ----------------------------------------------------------------

# 處理分類特徵
# 將 Sex 變為 1(female)跟 0(male)
for database in combine:
    database["Sex"] = database["Sex"].map({"female": 1, "male": 0}).astype(int)

# 將 Embarked 變為 0(S)、1(C)、2(Q)
for database in combine:
    database["Embarked"] = (
        database["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype(int)
    )

# 將 Title 變為 1、2、3、4、5
titleMapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for database in combine:
    # 將缺失值替換為 'Unknown'
    database["Title"] = database["Title"].fillna("Unknown")
    # 將 'Rare' 值視為未知的情況
    database["Title"] = (
        database["Title"]
        .map(lambda x: titleMapping[x] if x in titleMapping else 0)
        .astype(int)
    )

# ----------------------------------------------------------------

# 將連續特徵轉為離散特徵
# 將 Age 等分為五個區間
train_df["AgeBand"] = pd.cut(train_df["Age"], 5)
for database in combine:
    database.loc[database["Age"] < 16, "Age"] = 0
    database.loc[(database["Age"] >= 16) & (database["Age"] < 32), "Age"] = 1
    database.loc[(database["Age"] >= 32) & (database["Age"] < 48), "Age"] = 2
    database.loc[(database["Age"] >= 48) & (database["Age"] < 64), "Age"] = 3
    database.loc[database["Age"] >= 64, "Age"] = 4
train_df.drop(["AgeBand"], axis=1, inplace=True)
combine = [train_df, test_df]

# 將 Fare 依照數量分為五個區間
train_df["FareBand"] = pd.qcut(train_df["Fare"], 4)
test_df["Fare"] = test_df["Fare"].fillna(test_df["Fare"].mean())
for database in combine:
    database.loc[database["Fare"] <= 7.91, "Fare"] = 0
    database.loc[(database["Fare"] > 7.91) & (database["Fare"] <= 14.454), "Fare"] = 1
    database.loc[(database["Fare"] > 14.454) & (database["Fare"] <= 31), "Fare"] = 2
    database.loc[database["Fare"] > 31, "Fare"] = 3
    database["Fare"] = database["Fare"].astype(int)
train_df.drop(["FareBand"], axis=1, inplace=True)
combine = [train_df, test_df]

# ----------------------------------------------------------------

# 合併特徵（新增一列判斷是否為一個人）
for database in combine:
    database["Family"] = database["SibSp"] + database["Parch"] + 1

for database in combine:
    database["IsAlone"] = 1
    database.loc[database["Family"] > 1, "IsAlone"] = 0

train_df.drop(["SibSp", "Parch", "Family"], inplace=True, axis=1)
test_df.drop(["SibSp", "Parch", "Family"], inplace=True, axis=1)
combine = [train_df, test_df]

# ----------------------------------------------------------------

# 輸出處理完的資料
train_df.to_csv("Data/train_df_ready.csv", index=False)
test_df.to_csv("Data/test_df_ready.csv", index=False)
