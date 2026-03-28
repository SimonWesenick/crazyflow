import jax
import jax.numpy as jnp
from jax_mppi import mppi

# Define dynamics and cost functions
def dynamics(state, action):
    # Your dynamics model here
    return state + action

def running_cost(state, action):
    # Your cost function here
    return jnp.sum(state**2) + jnp.sum(action**2)

# Create configuration and initial state
config, mppi_state = mppi.create(
    nx=4, nu=2,
    noise_sigma=jnp.eye(2) * 0.1,
    horizon=20,
    lambda_=1.0
)

# Control loop
key = jax.random.PRNGKey(0)
current_obs = jnp.zeros(4)

# JIT compile the command function for performance
jitted_command = jax.jit(mppi.command, static_argnames=['dynamics', 'running_cost'])

for _ in range(100):
    key, subkey = jax.random.split(key)
    action, mppi_state = jitted_command(
        config,
        mppi_state,
        current_obs,
        dynamics=dynamics,
        running_cost=running_cost
    )
    # Apply action to environment...