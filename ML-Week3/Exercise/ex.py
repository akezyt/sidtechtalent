from IPython.display import Image
import matplotlib.pyplot as plt

import pandas as pd

df = pd.read_csv('bank/bank-additional-full.csv')

print(df.head())

from sklearn.preprocessing import LabelEncoder

y = df.y.values
le = LabelEncoder()
y = le.fit_transform(y)
print(le.classes_)

X = df

print(X.info())

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

X = df.iloc[:, 0:18]

month_encoder = LabelEncoder()

X = pd.get_dummies(X, columns=['month', 'day_of_week', 'poutcome', 'loan', 'marital', 'default', 'housing', 'education', 'job', 'contact'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
    train_test_split(X, y,
                     test_size=0.2,
                     stratify=y,
                     random_state=1)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipe_lr = make_pipeline(StandardScaler(),
                        LogisticRegression(random_state=1))

param_grid_lr = [{'logisticregression__C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}]

gs_lr = GridSearchCV(estimator=pipe_lr,
                  param_grid=param_grid_lr,
                  scoring='accuracy',
                  cv=10,
                  refit=True,
                    n_jobs=2)

gs_lr = gs_lr.fit(X_train, y_train)

print(gs_lr.best_score_)
print(gs_lr.best_params_)

pipe_dt = make_pipeline(DecisionTreeClassifier(random_state=1))

param_grid_dt = [{'decisiontreeclassifier__max_depth': [1, 2, 3, 4, 5, 6, 7, None],
               'decisiontreeclassifier__min_samples_leaf': [1, 2, 3, 4, 5, 6, 7]}]

gs_dt = GridSearchCV(estimator=pipe_dt,
                  param_grid=param_grid_dt,
                  scoring='accuracy',
                  cv=10,
                  refit=True)

gs_dt = gs_dt.fit(X_train, y_train)

print(gs_dt.best_score_)
print(gs_dt.best_params_)


best_lr = gs_lr.best_estimator_
best_dt = gs_dt.best_estimator_

y_proba_lr = best_lr.predict_proba(X_test)[:, 1]
y_proba_dt = best_dt.predict_proba(X_test)[:, 1]

from sklearn.metrics import roc_auc_score

"""
GET Y_PROBA
"""
print('LR AUC Score: %.3f' % roc_auc_score(y_true=y_test, y_score=y_proba_lr))
print('DT AUC Score: %.3f' % roc_auc_score(y_true=y_test, y_score=y_proba_dt))

best_dt = gs_dt.best_estimator_.fit(X, y)