import argparse
import time
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import jax.numpy as jnp
import jax.random as jrand
import equinox as eqx
import optax
from jax import vmap

from model import FFN
from train_step import make_step_adam_prox
from data_generator import dataloader
from altermin_schedular import allocate_model, collect_data_groups, batch_warmup
from metrics import RMSELoss,  AIC


def TestLoss(models, x_test, y_test):
    z = allocate_model(models, x_test, y_test)
    ytest_pred = np.array([]).reshape(0, 1)
    ytest_true = np.array([]).reshape(0, 1)
    for i in range(args.k):
        xi_, yi_ = collect_data_groups(i, x_test, y_test, z)
        yi_pred = vmap(models[i], in_axes=(0))(xi_)

        ytest_pred = jnp.concatenate([ytest_pred, yi_pred])
        ytest_true = jnp.concatenate([ytest_true, yi_])

    test_loss = RMSELoss(ytest_pred, ytest_true)

    return test_loss


parser = argparse.ArgumentParser()
parser.add_argument('--layer_sizes', '--list',
                    nargs='+',  type=int, default=[200, 30])
parser.add_argument('--data_classes', type=int, default=1)
parser.add_argument('--layer_nums', type=int)
parser.add_argument('--init_learn_rate', type=float, default=3e-4)
parser.add_argument('--adam_learn_rate', type=float, default=1e-3)
parser.add_argument('--adam_epsilon', type=float, default=1e-8)
parser.add_argument('--is_relu', type=int, default=1, choices=[0, 1])
parser.add_argument('--use_bias', type=str)
parser.add_argument('--ridge_param', type=float, default=0.1)
parser.add_argument('--lasso_param_ratio', type=float, default=0.1)
parser.add_argument('--group_lasso_param', type=float, default=0.1)
parser.add_argument('--decay', type=float, default=0.97)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--n_epochs', type=int, default=300)
parser.add_argument('--seed', type=int, default=20010218)
parser.add_argument('--num_p', type=int, default=200)
parser.add_argument('--num_groups', type=int, default=2) 
parser.add_argument('--n_train_obs', type=int, default=300)
parser.add_argument('--n_test_obs', type=int, default=100)
parser.add_argument('--balance', type=float, default=0.5)
parser.add_argument('--k', type=int, default=2)
parser.add_argument('--err_dist', type=int, default=0)
parser.add_argument('--round', type=int, default=1)
parser.add_argument('--linear', action='store_true')
parser.add_argument('--project', type=str)


args = parser.parse_args()

if args.linear:
    is_linear = 'linear'
else:
    is_linear = 'nonlinear'

if args.balance == 0.5:
    is_balance = 'balance'
else:
    is_balance = 'imbalance'


key = jrand.PRNGKey(args.seed)
loader_key, *model_keys = jrand.split(key, args.k + 1)


models: Sequence[FFN] = []
opt_states = []
optims = []
for i in range(args.k):
    model = FFN(
        layer_sizes=args.layer_sizes,
        data_classes=args.data_classes,
        is_relu=args.is_relu,
        layer_nums=args.layer_nums,
        use_bias=False,
        lasso_param_ratio=args.lasso_param_ratio,
        group_lasso_param=args.group_lasso_param,
        ridge_param=args.ridge_param,
        init_learn_rate=args.init_learn_rate,
        adam_learn_rate=args.adam_learn_rate,
        adam_epsilon=args.adam_epsilon,
        key=model_keys[i]
    )
    optim = optax.adam(args.adam_learn_rate, eps=args.adam_epsilon)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    models.append(model)
    opt_states.append(opt_state)
    optims.append(optim)

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

X = jnp.asarray(df.values[:, 0:100], dtype=jnp.float32)
y = jnp.asarray(df.values[:, 100].reshape(-1, 1), dtype=jnp.float32)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = args.adam_learn_rate
batch_size = x_train.shape[0]

layer1_size = args.layer_sizes[1] if len(args.layer_sizes) >= 2 else 0

start = time.time()

for step, (xi, yi) in zip(range(args.n_epochs), dataloader(
            [x_train, y_train], batch_size, key=loader_key)
    ):

    # if step == 0:
    #     z = batch_warmup(args.k, xi, yi)
    # else:
    z = allocate_model(models, xi, yi)
    y_pred = np.array([]).reshape(0, 1)
    y_true = np.array([]).reshape(0, 1)
    for i in range(args.k):
        xi_, yi_= collect_data_groups(i, xi, yi, z)
        yi_pred, all_loss, smooth_loss, unpen_loss, models[i], opt_states[i], lr = make_step_adam_prox(
                models[i], optims[i], opt_states[i], xi_, yi_, lr, decay=args.decay)
        y_pred = jnp.concatenate([y_pred, yi_pred])
        y_true = jnp.concatenate([y_true, yi_])

    train_loss = RMSELoss(y_pred, y_true)

    test_loss = TestLoss(models, x_test, y_test)

    if (step + 1) == args.n_epochs:
        end = time.time()
        supports = 0
        for g in range(args.k):
            support = models[g].support()
            supports += support
            # print(f"model{g} support: {support}")
        
        print(f"{args.k}, {train_loss}, {test_loss}, {AIC(train_loss, x_train.shape[0], supports, layer1_size)},{args.err_dist}, {args.n_train_obs}, {args.round}, {step + 1}, {end-start}")