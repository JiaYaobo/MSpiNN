from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans
import jax.numpy as jnp
from jax import vmap, jit

from model import FNN


@jit
def HuberLoss(y_pred, y_true):
    abs_err = jnp.abs(y_pred - y_true)
    delta = 1.4826 * jnp.mean(abs_err) * 1.345
    loss = jnp.where(abs_err < delta, 
                    1/2 * abs_err ** 2, 
                    delta * abs_err - 1/2 * delta ** 2)
    return loss


def batch_warmup(n_groups, x, y):
    kms = KMeans(n_clusters=n_groups)
    data = np.hstack([x, y])
    kms.fit(data)
    return jnp.array(kms.labels_)


def allocate_model(models: Sequence[FNN], x, y):
    loss = []
    for i in range(len(models)):
        y_pred = vmap(models[i], in_axes=(0))(x)
        huber_loss = HuberLoss(y_pred, y)
        loss.append(huber_loss)


    loss = jnp.squeeze(jnp.asarray(loss), axis=2).T
    index = jnp.argmin(loss, axis=1)
    return index


def collect_data_groups(which_group, x, y, z):
    return x[z == which_group, ], y[z == which_group, ]