import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx

from model import FFN
from loss import all_pen_loss, mse_loss


@eqx.filter_jit
def clip_gradient(grads):
    leaves, treedef = jtu.tree_flatten(
        grads, is_leaf=lambda x: isinstance(x, eqx.nn.Linear)
    )
    new_leaves = []
    for leaf in leaves:
        if isinstance(leaf, eqx.nn.Linear):
            lim = 0.01
            leaf = eqx.tree_at(
                lambda x: x.weight, leaf, leaf.weight.clip(-lim, lim)
            )
        new_leaves.append(leaf)
    return jtu.tree_unflatten(treedef, new_leaves)


@eqx.filter_jit
def make_step(model: FFN, optim, opt_state, x, y, lr=None, decay=None):
    pred, _, __, ___, grads = mse_loss(
        model, x, y)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return pred, None, None, None, model, opt_state, lr


@eqx.filter_jit
def make_step_adam_prox(model: FFN, optim, opt_state, x, y, lr=0.001, decay=0.98):
    pred, all_loss, smooth_loss, unpen_loss, grads = all_pen_loss(
        model, x, y)
    updates, opt_state = optim.update(grads, opt_state)
    # Do proximal gradient step
    values, treedef = jtu.tree_flatten(updates)
    # adam_weights = jnp.abs(values[0])
    sz = lr * jnp.sqrt(1 - 0.999) / (1 - 0.9)
    adam_weights = jtu.tree_flatten(opt_state[0].adam_weights)[0][0] * sz
    # Do proximal gradient for lasso: soft threshold
    weights = model.layers[0].weight
    weights_updated = jnp.multiply(jnp.sign(weights), jnp.maximum(
        jnp.abs(weights) - model.lasso_param * adam_weights, 0))

    # Do proximal gradient step for group lasso
    group_norms = 1e-10 + jnp.linalg.norm(weights, axis=1).reshape(-1, 1)
    group_lasso_scale_factor = jnp.maximum(
        1 - model.group_lasso_param * adam_weights / group_norms, 0)

    weights_updated = jnp.multiply(group_lasso_scale_factor, weights_updated)

    # update model
    values[0] = weights_updated - weights
    updates = jtu.tree_unflatten(treedef, values)
    model = eqx.apply_updates(model, updates)

    return pred, all_loss, smooth_loss, unpen_loss, model, opt_state, lr


@eqx.filter_jit
def make_step_prox(model: FFN, optim, opt_state, x, y, lr=None, decay=0.98):
    pred, all_loss, smooth_loss, unpen_loss, grads = all_pen_loss(
        model, x, y)
    updates = jtu.tree_map(lambda g: - model.lasso_param * lr * g, grads)
    # model = eqx.apply_updates(model, updates)
    # do proximal gradient step
    values, treedef = jtu.tree_flatten(updates)

    # Do proximal gradient for lasso: soft threshold
    weights = model.layers[0].weight
    weights += values[0]
    weigths_updated = jnp.multiply(jnp.sign(weights), jnp.maximum(
        jnp.abs(weights) - model.lasso_param * lr, 0))

    # do proximal gradient step for group lasso
    group_norms = 1e-10 + jnp.linalg.norm(weights, axis=1).reshape(-1, 1)
    group_lasso_scale_factor = jnp.maximum(
        1 - model.group_lasso_param * lr / group_norms, 0)
    weights_updated = jnp.multiply(group_lasso_scale_factor, weigths_updated)

    values[0] = weights_updated - model.layers[0].weight
    updates = jtu.tree_unflatten(treedef, values)
    model = eqx.apply_updates(model, updates)

    lr = lr * decay

    return pred, all_loss, smooth_loss, unpen_loss, model, opt_state, lr
