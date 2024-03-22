import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# 找到 PassengerId
test = pd.read_csv("Data/test.csv")
PassengerId = test["PassengerId"]

# 導入訓練集與測試集（處理過的）
train_df = pd.read_csv("Data/train_df_ready.csv")
test_df = pd.read_csv("Data/test_df_ready.csv")
combine = [train_df, test_df]

# 訓練特徵（X_train）和目標變數（Y_train）
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]

# 使用決策樹
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(test_df)

# 準確率達 86.42
conclusion = pd.DataFrame({"PassengerId": PassengerId, "Survived": Y_pred})
conclusion.to_csv("Data/conclusion .csv", index=False)
