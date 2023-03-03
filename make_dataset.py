import argparse

import numpy as np
import pandas as pd

from data_generator import get_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=20010218)
parser.add_argument('--num_p', type=int, default=200)
parser.add_argument('--num_groups', type=int, default=2) 
parser.add_argument('--n_train_obs', type=int, default=300)
parser.add_argument('--n_test_obs', type=int, default=100)
parser.add_argument('--balance', type=float, default=0.5)
parser.add_argument('--err_dist', type=int, default=0)
parser.add_argument('--linear', action='store_true')
parser.add_argument('--train', action='store_true')

args = parser.parse_args()

if args.linear:
    from linear_func import func1, func2
else:
    from nonlinear_func import func1, func2


is_linear = None
is_balance = None 
is_train = None
n = None

if args.linear:
    is_linear = 'linear'
else:
    is_linear = 'nonlinear'


if args.balance == 0.5:
    is_balance = 'balance'
else:
    is_balance = 'imbalance'

if args.train:
    is_train = 'train'
    n = args.n_train_obs
else:
    is_train = 'test'
    n = args.n_test_obs



if args.linear:
    data_file = './data/linear/'+is_linear+'_'+is_train+'_'+is_balance+'_'+str(n)+'_err'+str(args.err_dist)+'.csv'
else:
    data_file = './data/nonlinear/'+is_linear+'_'+is_train+'_'+is_balance+'_'+str(n)+'_err'+str(args.err_dist)+'.csv'

x_train, y_train, group_train = get_dataset(args.num_p, args.num_groups, 
                    [int(args.n_train_obs*args.balance), int(args.n_train_obs * (1 - args.balance))], args.err_dist, func_list=[func1, func2])


data = np.hstack([x_train, y_train, group_train])
df = pd.DataFrame(data)
df.to_csv(data_file)