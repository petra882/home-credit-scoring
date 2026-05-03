import numpy as np
import polars as pl
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier
import seaborn as sns
from catboost import Pool
from matplotlib import pyplot as plt


main_ds = pl.read_csv("ds/application_train.csv", try_parse_dates=True)

result = main_ds.select(
    (pl.col("DAYS_BIRTH")*-1//365).alias("age_years"),
    pl.col("NAME_CONTRACT_TYPE"),
    (pl.col("AMT_CREDIT")/pl.col("AMT_INCOME_TOTAL")).alias("credit_to_income"),
    pl.col("EXT_SOURCE_1"),
    pl.col("EXT_SOURCE_2"),
    pl.col("EXT_SOURCE_3"),
    pl.col("TARGET"),
    pl.col("NAME_INCOME_TYPE"),
    pl.col("CODE_GENDER")
).drop_nulls()
result_to_encode = result.select(
    pl.col("NAME_CONTRACT_TYPE"),
    pl.col("NAME_INCOME_TYPE"),
    pl.col("CODE_GENDER")
).to_numpy()

numeric_cols = ["age_years", "credit_to_income", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
categorical_cols = ["NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE", "CODE_GENDER"]

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_categorial_encoded = encoder.fit_transform(result_to_encode)
X_numeric = result.drop(["NAME_CONTRACT_TYPE", "TARGET", "NAME_INCOME_TYPE", "CODE_GENDER"]).to_numpy()
y = result["TARGET"].to_numpy()
X_result = np.hstack([X_numeric, X_categorial_encoded])
feature_names = numeric_cols + list(encoder.get_feature_names_out(categorical_cols))
X_train, X_test, y_train, y_test = train_test_split(X_result, y, test_size=0.2, random_state=42)

"""
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
y_pred = dummy_clf.predict(X_test) #roc-auc = 0.5
accuracy - 0.92
"""

"""
rnd_forest = RandomForestClassifier(class_weight="balanced", random_state=42)
rnd_forest.fit(X_train, y_train)
y_pred_proba = rnd_forest.predict_proba(X_test)[:, 1]
print("RandomForest ROC-AUC:", roc_auc_score(y_test, y_pred_proba))
y_pred = rnd_forest.predict(X_test)
print(classification_report(y_test, y_pred))
roc-auc = 0.71, recall - 0.02 
"""

model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    scale_pos_weight=11.5,
    random_seed=42,
    verbose=100
)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
pred_proba = model.predict_proba(X_test)[:, 1]
print(roc_auc_score(y_test, pred_proba))
print(classification_report(y_test, prediction))

feature_importance = model.get_feature_importance()


importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=True)


top_features = importance_df.tail(20)

plt.figure(figsize=(10, 8))
plt.barh(top_features['feature'], top_features['importance'])
plt.xlabel('Важность')
plt.title('Feature Importance (CatBoost)')
plt.tight_layout()
plt.show()