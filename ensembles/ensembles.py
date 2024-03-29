import argparse

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LassoCV

from metrics import RMSELoss

rf = RandomForestRegressor()
ar = AdaBoostRegressor()
br = BaggingRegressor()
gbr = GradientBoostingRegressor()
etr = ExtraTreesRegressor()
mlp = MLPRegressor()
lasso = LassoCV()

parser = argparse.ArgumentParser()
parser.add_argument('--decay', type=float, default=0.97)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--n_epochs', type=int, default=400)
parser.add_argument('--seed', type=int, default=20010218)
parser.add_argument('--num_p', type=int, default=100)
parser.add_argument('--num_groups', type=int, default=2)
parser.add_argument('--n_train_obs', type=int, default=300)
parser.add_argument('--n_test_obs', type=int, default=100)
parser.add_argument('--balance', type=float, default=0.5)
parser.add_argument('--k', type=int, default=2)
parser.add_argument('--err_dist', type=int, default=0)
parser.add_argument('--round', type=int, default=1)
parser.add_argument('--linear', action='store_true')
parser.add_argument('--project', type=str, default='ensemble.csv')

args = parser.parse_args()

if args.linear:
    is_linear = 'linear'
else:
    is_linear = 'nonlinear'

if args.balance == 0.5:
    is_balance = 'balance'
else:
    is_balance = 'imbalance'

train_data_file = '../data/' + is_linear + '/' + is_linear + '_train_' + is_balance + '_' + str(
    args.n_train_obs) + '_err' + str(args.err_dist) + '.csv'
test_data_file = '../data/' + is_linear + '/' + is_linear + '_test_' + is_balance + '_' + str(
    args.n_test_obs) + '_err' + str(args.err_dist) + '.csv'

train_df = pd.read_csv(train_data_file)
test_df = pd.read_csv(test_data_file)

x_train = train_df.iloc[:, 1:(args.num_p + 1)].values
y_train = train_df.iloc[:, 101].values.reshape(-1, 1).ravel()
group_train = train_df.iloc[:, 102].values.reshape(-1, 1)

x_test = test_df.iloc[:, 1:(args.num_p + 1)].values
y_test = test_df.iloc[:, 101].values.reshape(-1, 1).ravel()
group_test = test_df.iloc[:, 102].values.reshape(-1, 1)

rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

ar.fit(x_train, y_train)
y_pred_ar = ar.predict(x_test)

br.fit(x_train, y_train)
y_pred_br = br.predict(x_test)

etr.fit(x_train, y_train)
y_pred_etr = etr.predict(x_test)

gbr.fit(x_train, y_train)
y_pred_gbr = gbr.predict(x_test)

print(f"{RMSELoss(y_test, y_pred_rf)}, {RMSELoss(y_test, y_pred_ar)}, "
      f"{RMSELoss(y_test, y_pred_br)}, {RMSELoss(y_test, y_pred_etr)}, {RMSELoss(y_test, y_pred_gbr)}, ")
