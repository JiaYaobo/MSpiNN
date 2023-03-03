import jax.numpy as jnp
from jax  import jit

@jit
def MSELoss(y_pred, y_true):
    return jnp.mean((y_pred - y_true) ** 2)

@jit
def RMSELoss(y_pred, y_true):
    return jnp.sqrt(jnp.mean((y_pred - y_true) ** 2))

def BIC(y_pred, y_true, num_p, k, n, supports):
    return jnp.log(RMSELoss(y_pred, y_true)) + \
          jnp.log(jnp.log(num_p * k)) * jnp.log(n)/n * supports
