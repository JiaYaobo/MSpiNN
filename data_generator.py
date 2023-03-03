import numpy as np
import jax.numpy as jnp
import jax.random as jrand


class Dataset:
    def __init__(self, x, y, y_true=None, group=None):
        self.x = x
        self.y = y
        self.y_true = y_true
        self.group = group

    def create_restricted(self, max_relevant_idx):
        return Dataset(
                self.x[:, :max_relevant_idx],
                self.y,
                self.y_true)

class DataGenerator:
    def __init__(self, num_p, func, dist, group=0):
        self.num_p = num_p
        self.func = func
        self.group=group
        self.dist = dist

    def create_data(self, n_obs, xs):
        assert n_obs > 0
        true_ys = self.func(xs)
        true_ys = np.reshape(true_ys, (true_ys.size, 1))
        if self.dist == 0:
            eps = 1.0 * np.random.randn(n_obs, 1).reshape(n_obs, 1) + 0.0 * np.random.standard_t(1, n_obs).reshape(n_obs, 1)
        elif self.dist == 1:
            eps = .7 * np.random.randn(n_obs, 1).reshape(n_obs, 1) + .3 * np.random.standard_t(1, n_obs).reshape(n_obs, 1)
        else:
            eps = 0 * np.random.randn(n_obs, 1).reshape(n_obs, 1) + 1.0 * np.random.standard_t(1, n_obs).reshape(n_obs, 1)
        y = true_ys + 0.5 * eps
        true_ys = self.func(xs)
        true_ys = np.reshape(true_ys, (true_ys.size, 1))
        y = np.array(np.random.random_sample((true_ys.size, 1)) < true_ys, dtype=int)
        return Dataset(xs, y, true_ys, group=self.group)

def get_dataset(num_p, num_groups, n_obs, err_dist, func_list):
    if isinstance(n_obs, int):
        n_obs = [n_obs] * num_groups
    x = np.array([]).reshape(0, num_p)
    y = np.array([]).reshape(0, 1)
    group = np.array([]).reshape(0, 1)


    p = num_p
    mean = np.zeros(p)

    cov = np.zeros((p, p))
    for j in range(p):
        for k in range(p):
            cov[j, k] = 0.5 ** abs(j - k)


    for i in range(num_groups):
        xs = np.random.multivariate_normal(mean, cov, n_obs[i])
        dt = DataGenerator(num_p, func_list[i], err_dist, i).create_data(n_obs[i], xs)
        x = np.vstack([dt.x, x])
        y = np.vstack([dt.y, y])
        group_ = np.ones((dt.x.shape[0], 1)) * i
        group = np.vstack([group_, group])

    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()


    return x, y, group


def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jrand.permutation(key, indices)
        (key,) = jrand.split(key, 1)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size