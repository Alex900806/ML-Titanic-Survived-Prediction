import pandas as pd
import matplotlib.pyplot as plt

# 目的：查看所有特徵跟存活與否的關聯程度
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# 導入訓練集與測試集
train_df = pd.read_csv("Data/train.csv")
test_df = pd.read_csv("Data/test.csv")
combine = [train_df, test_df]

# Pclass 與 Survived 關係
# 艙位等級越高 存活率越高
result = (
    train_df[["Pclass", "Survived"]]
    .groupby(["Pclass"], as_index=False)
    .mean()
    .sort_values(by="Survived", ascending=False)
    .to_string(index=False)
)
# print(result)

# Sex 與 Survived 關係
# 女性存活率明顯高於男性
result = (
    train_df[["Sex", "Survived"]]
    .groupby(["Sex"], as_index=False)
    .mean()
    .sort_values(by="Survived", ascending=False)
    .to_string(index=False)
)
# print(result)

# SibSp 與 Survived 關係
# 擁有越少兄弟姐妹（Siblings）和配偶（Spouses）的存活率越高
result = (
    train_df[["SibSp", "Survived"]]
    .groupby(["SibSp"], as_index=False)
    .mean()
    .sort_values(by="Survived", ascending=False)
    .to_string(index=False)
)
# print(result)

# Parch 與 Survived 關係
# 擁有越少（少於三個）家人的存活率越高
result = (
    train_df[["Parch", "Survived"]]
    .groupby(["Parch"], as_index=False)
    .mean()
    .sort_values(by="Survived", ascending=False)
    .to_string(index=False)
)
# print(result)

# Embarked 與 Survived 關係
# 在 C 上船的存活率最高
result = (
    train_df[["Embarked", "Survived"]]
    .groupby(["Embarked"], as_index=False)
    .mean()
    .sort_values(by="Survived", ascending=False)
    .to_string(index=False)
)
# print(result)

# Age 與 Survived 關係
# 可以看出 20~40 歲得存活率較低
fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.hist(train_df.Age[train_df["Survived"] == 0], bins=20)
ax1.set_title("死亡率年齡分佈")
ax1.set_xlabel("年齡")
ax1.set_yticks([10, 20, 30, 40, 50, 60])

ax2.hist(train_df.Age[train_df["Survived"] == 1], bins=20)
ax2.set_title("存活率年齡分佈")
ax2.set_xlabel("年齡")
ax2.set_yticks([10, 20, 30, 40, 50, 60])
plt.show()

# Fare 與 Survived 關係
# 較難看出票票價跟存活率的關係
fare = train_df.Fare[train_df["Survived"] == 1]
plt.hist(fare)
plt.title("存活人數")
plt.xlabel("票價")
plt.show()
