# --- Imports ---
import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- Settings ---
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

# --- Load Data ---
df = pd.read_csv('data/Training.csv')
te = pd.read_csv('data/Testing.csv')

# Drop unnamed column
df = df.drop(df.columns[133], axis=1)

# --- Split Data ---
x = df.drop(columns='prognosis')
y = df['prognosis']
x_te = te.drop(columns='prognosis')
y_te = te['prognosis']

# --- Model Without PCA ---
rf = RandomForestClassifier()
params = {
    'criterion': ['gini', 'entropy'],
    'min_samples_split': list(np.arange(2, 31)),
    'min_samples_leaf': list(np.arange(2, 51)),
    'n_estimators': [10]
}
nrf = RandomizedSearchCV(rf, param_distributions=params, cv=10, n_jobs=-1, scoring='accuracy', random_state=20)
nrf.fit(x, y)
nrf_no_pca = nrf.best_estimator_

# --- Evaluate Without PCA ---
pred_train = nrf_no_pca.predict(x)
pred_test = nrf_no_pca.predict(x_te)

at = [accuracy_score(y, pred_train)]
a = [accuracy_score(y_te, pred_test)]
pt = [precision_score(y, pred_train, average="weighted")]
p = [precision_score(y_te, pred_test, average="weighted")]
rt = [recall_score(y, pred_train, average="weighted")]
r = [recall_score(y_te, pred_test, average="weighted")]
f1t = [f1_score(y, pred_train, average="weighted")]
f1 = [f1_score(y_te, pred_test, average="weighted")]
crt = [classification_report(y, pred_train)]
cr = [classification_report(y_te, pred_test)]
cmt = [confusion_matrix(y, pred_train)]
cm = [confusion_matrix(y_te, pred_test)]

# --- PCA Transformation ---
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_pca = PCA(n_components=65).fit_transform(x_scaled)

x_te_scaled = scaler.transform(x_te)
x_te_pca = PCA(n_components=65).fit(scaler.fit_transform(x)).transform(x_te_scaled)

# --- Model With PCA ---
nrf = RandomizedSearchCV(rf, param_distributions=params, cv=10, n_jobs=-1, scoring='accuracy', random_state=20)
nrf.fit(x_pca, y)
nrf_pca = nrf.best_estimator_

# --- Evaluate With PCA ---
pred_train_pca = nrf_pca.predict(x_pca)
pred_test_pca = nrf_pca.predict(x_te_pca)

at.append(accuracy_score(y, pred_train_pca))
a.append(accuracy_score(y_te, pred_test_pca))
pt.append(precision_score(y, pred_train_pca, average="weighted"))
p.append(precision_score(y_te, pred_test_pca, average="weighted"))
rt.append(recall_score(y, pred_train_pca, average="weighted"))
r.append(recall_score(y_te, pred_test_pca, average="weighted"))
f1t.append(f1_score(y, pred_train_pca, average="weighted"))
f1.append(f1_score(y_te, pred_test_pca, average="weighted"))
crt.append(classification_report(y, pred_train_pca))
cr.append(classification_report(y_te, pred_test_pca))
cmt.append(confusion_matrix(y, pred_train_pca))
cm.append(confusion_matrix(y_te, pred_test_pca))

# --- Print Reports ---
mods = ['without_pca', 'with_pca']
for m, c in zip(mods, cr):
    print(f"{m} Test Classification Report:\n{c}\n")

# --- Save the Best Model (with PCA) ---
with open("model/nabeeh_model.pkl", "wb") as f:
    pickle.dump(nrf_pca, f)

print("✅ Model saved to model/nabeeh_model.pkl")
