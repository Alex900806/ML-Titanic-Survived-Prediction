import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# 目的：找到最佳測試模型


# 導入訓練集與測試集（處理過的）
train_df = pd.read_csv("Data/train_df_ready.csv")
test_df = pd.read_csv("Data/test_df_ready.csv")
combine = [train_df, test_df]

# 訓練特徵（X_train）和目標變數（Y_train）
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]

# ----------------------------------------------------------------

# 邏輯回歸
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(test_df)
acc_log = round(logreg.score(X_train, Y_train) * 100, 3)

# ----------------------------------------------------------------

# SVM
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(test_df)
acc_svc = round(svc.score(X_train, Y_train) * 100, 3)

# ----------------------------------------------------------------

# 決策樹
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(test_df)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 3)

# ----------------------------------------------------------------

# 隨機森林
random_forest = RandomForestClassifier()
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(test_df)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 3)


# ----------------------------------------------------------------

# KNN
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
Y_pred = knn.predict(test_df)
acc_knn = round(knn.score(X_train, Y_train) * 100, 3)

# ----------------------------------------------------------------

# Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(test_df)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 3)

# ----------------------------------------------------------------

# 比較各個模型
models = pd.DataFrame(
    {
        "Models": [
            "Logistic Regression",
            "Support Vector Machine",
            "Decision Tree",
            "Random Forest",
            "KNN",
            "Naive Bayes",
        ],
        "Score": [
            acc_log,
            acc_svc,
            acc_decision_tree,
            acc_random_forest,
            acc_knn,
            acc_gaussian,
        ],
    }
)
result = models.sort_values(by="Score", ascending=False).reset_index(drop=True)
print(result)
#                    Models   Score
# 0           Decision Tree  86.420
# 1           Random Forest  86.420
# 2  Support Vector Machine  83.389
# 3                     KNN  82.716
# 4     Logistic Regression  80.808
# 5             Naive Bayes  77.104

# 最佳模型是 Decision Tree 跟 Random Forest
