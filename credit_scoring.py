#!/usr/bin/env python
# coding: utf-8

# === Import libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import unique_labels

# === Function to plot confusion matrix ===
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Greys):
    if not title:
        title = 'Normalized confusion matrix' if normalize else 'Confusion matrix'

    cm = confusion_matrix(y_true, y_pred)
    classes = np.array(classes)  # FIXED: removed indexing by label values

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

# === Load dataset ===
dataset = pd.read_csv("/Users/preetbarot/Desktop/credit scoring model/dataset/estadistical.csv")

# === Feature and target separation ===
x = dataset.drop("Receive/ Not receive credit ", axis=1)
y = dataset["Receive/ Not receive credit "]

# === Encode categorical columns ===
cat_mask = x.dtypes == object
cat_cols = x.columns[cat_mask].tolist()
le = preprocessing.LabelEncoder()
x[cat_cols] = x[cat_cols].apply(lambda col: le.fit_transform(col))

# === Train-test split ===
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)

# === Feature scaling ===
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

# === KNN Model ===
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(xtrain, ytrain)
pred_knn = knn.predict(xtest)
print("KNN Accuracy:", accuracy_score(ytest, pred_knn))

# === Random Forest ===
forest = RandomForestClassifier(max_depth=2, random_state=0)
forest.fit(xtrain, ytrain)
pred_forest = forest.predict(xtest)
print("Random Forest Accuracy:", accuracy_score(ytest, pred_forest))

# === Logistic Regression ===
logreg = LogisticRegression(random_state=0, class_weight="balanced", max_iter=1000)
logreg.fit(xtrain, ytrain)
pred_logreg = logreg.predict(xtest)
print("Logistic Regression Accuracy:", accuracy_score(ytest, pred_logreg))

# === Confusion Matrix for Logistic Regression ===
plot_confusion_matrix(ytest, pred_logreg, classes=np.unique(y), normalize=False,
                      title='Logistic Regression Confusion Matrix')
plt.show()

# === Support Vector Machine ===
svm = SVC(gamma='auto', class_weight="balanced")
svm.fit(xtrain, ytrain)
pred_svm = svm.predict(xtest)
print("SVM Accuracy:", accuracy_score(ytest, pred_svm))
