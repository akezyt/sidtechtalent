import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

customers = pd.read_csv('Ecommerce Customers')

y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=10)

lm = LinearRegression()

lm.fit(X_train, y_train)

print(lm.intercept_)
print(lm.coef_)

coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])

print(coeff_df)

predictions = lm.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
