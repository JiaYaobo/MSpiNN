import jax.numpy as jnp
from jax  import jit

@jit
def MSELoss(y_pred, y_true):
    return jnp.mean((y_pred - y_true) ** 2)

@jit
def RMSELoss(y_pred, y_true):
    return jnp.sqrt(jnp.mean((y_pred - y_true) ** 2))

def AIC(loss, n, supports, upper_size):
    return jnp.log(loss**2) + supports * upper_size / n

def BIC(loss, num_p, k, n, supports, upper_size):
    return jnp.log(loss**2) + \
          jnp.log(jnp.log(num_p * k)) * jnp.log(n)/n * supports * upper_size


