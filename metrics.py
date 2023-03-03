import jax.numpy as jnp
from jax  import jit

@jit
def MSELoss(y_pred, y_true):
    return jnp.mean((y_pred - y_true) ** 2)

@jit
def RMSELoss(y_pred, y_true):
    return jnp.sqrt(jnp.mean((y_pred - y_true) ** 2))
