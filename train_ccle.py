import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV



fn_x = './data/CCLE/expression.csv'
fn_y = './data/CCLE/drug.csv'

def load_ccle_data():
    df_x = pd.read_csv(fn_x)
    df_y = pd.read_csv(fn_y)
    x = df_x.values[:,1:101]
    y = np.expand_dims(df_y['Paclitaxel_ActArea'].values, axis=1)
    data = np.hstack([x, y])
    df = pd.DataFrame(data)
    return df

df = load_ccle_data().dropna(axis=0)

X = df.values[:,0:100]
y = df.values[:,100]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# for i in range(0,10):

#     rfr_pred = RandomForestRegressor().fit(X_train, y_train).predict(X_test)
#     abr_pred = AdaBoostRegressor().fit(X_train, y_train).predict(X_test)
#     gbr_pred = GradientBoostingRegressor().fit(X_train, y_train).predict(X_test)
#     lasso_pred = LassoCV().fit(X_train, y_train).predict(X_test)

#     print(f"{np.sqrt(np.mean((y_test - rfr_pred) ** 2))}, {np.sqrt(np.mean((y_test - abr_pred) ** 2))}, {np.sqrt(np.mean((y_test - gbr_pred) ** 2))}, {np.sqrt(np.mean((y_test - lasso_pred) ** 2))}")
