from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
import numpy as np

df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 'python-machine-learning-book-2nd-edition'
                 '/master/code/ch10/housing.data.txt',
                 header=None,
                 sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()

# select all 13 features
X = df.ix[:, :-1].values
y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

sc_x = StandardScaler()
x_std = sc_x.fit_transform(X_train)

"""
TODO:
Rescale X with StandardScaler
train Ridge regression with regularization parameter = 1.0 on X_train data
and then predict on the X_train and X_test data to obtain y_train_pred and y_test_pred
y_train_pred
y_test_pred
"""
reg_param = 1.0

ridge = Ridge(alpha=reg_param)
lasso = Lasso(alpha=reg_param)

# fit the model
ridge.fit(x_std, y_train)
lasso.fit(x_std, y_train)

ridge_y_train_pred = ridge.predict(sc_x.fit_transform(X_train))
lasso_y_train_pred = lasso.predict(sc_x.fit_transform(X_train))

print("ridge coef: "+ str(ridge.coef_))

ridge_y_test_pred = ridge.predict(sc_x.fit_transform(X_test))
lasso_y_test_pred = lasso.predict(sc_x.fit_transform(X_test))


from sklearn.metrics import mean_squared_error

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, ridge_y_train_pred),
        mean_squared_error(y_test, ridge_y_test_pred)))

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, lasso_y_train_pred),
        mean_squared_error(y_test, lasso_y_test_pred)))
