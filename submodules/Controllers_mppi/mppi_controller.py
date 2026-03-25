from dataclasses import dataclass, replace
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from .types import DynamicsFn, RunningCostFn, TerminalCostFn

@dataclass(frozen=True)
class MPPIConfig:
    #control param

    num_samples: int #K
    horizon: int #T
    lambda_: int #temperature
    nx: int
    nu: int
    lambda_: float
    u_scale: float
    u_per_command: int
    step_dependent_dynamics: bool
    rollout_samples: int  # M
    rollout_var_cost: float
    rollout_var_discount: float
    sample_null_action: bool
    noise_abs_cost: bool


@register_pytree_node_class
@dataclass
class MPPIState:
    U: jax.Array  # (T, nu) aktuelle nominale Kontrollsequenz
    u_init: jax.Array  # (nu,) shift
    noise_mu: jax.Array  
    noise_sigma: jax.Array  
    noise_sigma_inv: jax.Array
    u_min: Optional[jax.Array] #bounds
    u_max: Optional[jax.Array]
    key: jax.Array  # Pseudo Random Number Generator in JAX
    # zb key = jax.random.PRNGKey(0)
    #noise = jax.random.normal(key, shape=(10,))

    def tree_flatten(self):  #Jax tree: state besteh aus genau 9 teilen
        return (
            (
                self.U,
                self.u_init,
                self.noise_mu,
                self.noise_sigma,
                self.noise_sigma_inv,
                self.u_min,
                self.u_max,
                self.key,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

#Eingangsgrenzen     
def _bound_action(    
    action: jax.Array, u_min: Optional[jax.Array], u_max: Optional[jax.Array]
    ) -> jax.Array: #Eingangsgrenzen 
        if u_min is None and u_max is None:
            return action
        if u_min is None:
            assert u_max is not None
            return jnp.minimum(action, u_max)
        if u_max is None:
            return jnp.maximum(action, u_min)
        return jnp.clip(action, u_min, u_max)

#Eingangsgrenzen skalieren
def _scaled_bounds(
    u_min: Optional[jax.Array],
    u_max: Optional[jax.Array],
    u_scale: float,
) -> Tuple[Optional[jax.Array], Optional[jax.Array]]:
    if u_scale == 1.0 or u_scale == 0.0:
        return u_min, u_max
    u_min_scaled = None if u_min is None else (u_min / u_scale)
    u_max_scaled = None if u_max is None else (u_max / u_scale)
    return u_min_scaled, u_max_scaled

#opt folger-> führe erste Aktion aus -> für nächsten Zeitpunkt neuer Horizont der Länge T
# --> brauche shift
def _shift_nominal(mppi_state: MPPIState, shift_steps: int) -> MPPIState:
    if shift_steps <= 0:
        return mppi_state
    horizon = mppi_state.U.shape[0] #Anzahl Zeitschritte
    shift_steps = int(min(shift_steps, horizon)) #nie mehr Schritte als Arraylänge
    u_control = jnp.roll(mppi_state.U, -shift_steps, axis=0) #alles nach vorne verschieben
    fill = jnp.tile(mppi_state.u_init, (shift_steps, 1))
    u_control = u_control.at[-shift_steps:].set(fill) #setze letzte zeile auf u_init
    return replace(mppi_state, U=u_control)

#noise
def _sample_noise(
        key: jax.Array,
        num_samples: int,
        horizon: int,
        noise_mu: jax.Array,
        noise_sigma: jax.Array,
        sample_null_action: bool,
        ) -> Tuple[jax.Array, jax.Array]:
        key, subkey = jax.random.split(key) # ziehe Zufallszahlen, gebe key als neuen hauptkey zurück

        #beachte dim
        noise = jax.random.multivariate_normal(
            subkey,
            mean=noise_mu,
            cov=noise_sigma,
            shape=(num_samples, horizon),
        )

        if sample_null_action:
            noise = noise.at[0].set(jnp.zeros((horizon, noise_mu.shape[0])))
        return noise, key

def _state_for_cost(state: jax.Array, nx: int) -> jax.Array:
    if state.shape[-1] <= nx:
        return state
    return state[..., :nx]

#punktmasse? quadrotor dyn?
def _call_dynamics(
    dynamics: DynamicsFn,
    state: jax.Array,
    action: jax.Array,
    t: int,
    step_dependent: bool,
) -> jax.Array:
    if step_dependent:
        return dynamics(state, action, t)
    return dynamics(state, action)


def _call_running_cost(
    running_cost: RunningCostFn,
    state: jax.Array,
    action: jax.Array,
    t: int,
    step_dependent: bool,
) -> jax.Array:
    if step_dependent:
        return running_cost(state, action, t)
    return running_cost(state, action)


def _single_rollout_costs(
    config: MPPIConfig,
    current_obs: jax.Array,
    actions: jax.Array,
    dynamics: DynamicsFn,
    running_cost: RunningCostFn,
    terminal_cost: Optional[TerminalCostFn],
) -> Tuple[jax.Array, jax.Array]:
    
    def step_fn(state, inputs):
        t, action = inputs
        #x_k+1= f(x,u,(t))
        next_state = _call_dynamics(
            dynamics, state, action, t, config.step_dependent_dynamics
        )
        #reduz. auf erste nx komponenten (Kostenrelevanter Teil)
        cost_state = _state_for_cost(state, config.nx)
        #laufende kosten
        step_cost = _call_running_cost(
            running_cost, cost_state, action, t, config.step_dependent_dynamics
        )
        return next_state, step_cost
    #state = current_obs
    #step_costs = []

    #for t in range(horizon):
        #action = actions[t]
        #next_state = dynamics(state, action)
        #cost = running_cost(state, action)
        #step_costs.append(cost)
        #state = next_state

        #final_state = state

    ts = jnp.arange(config.horizon)
    #benutze scan statt for loop für jax comp.

    final_state, step_costs = jax.lax.scan(step_fn, current_obs, (ts, actions))
    if terminal_cost is None:
        terminal = jnp.array(0.0) #phi =0
    else: #phi = phi(xT,uT-1)
        terminal_state = _state_for_cost(final_state, config.nx)
        terminal = terminal_cost(terminal_state, actions[-1])
    return step_costs, terminal

#jetzt für jedes sample k ein rollout kostenwert

def _compute_rollout_costs(
    config: MPPIConfig,
    current_obs: jax.Array,
    actions: jax.Array,
    dynamics: DynamicsFn,
    running_cost: RunningCostFn,
    terminal_cost: Optional[TerminalCostFn],
) -> jax.Array: #parallel über jax
    per_step_costs, terminal_costs = jax.vmap(
        lambda a: _single_rollout_costs(
            config, current_obs, a, dynamics, running_cost, terminal_cost
        )
    )(actions) #actions -> stack der sequenzen

    mean_step_costs = per_step_costs
    var_step_costs = jnp.zeros_like(per_step_costs)

    if config.rollout_samples > 1:
        # Placeholder: Wenn man pro Aktionssequenz mehrere 
        # stochastische Rollouts hätte, könnte man pro Zeitschritt 
        # z. B. Mittelwertoder Varianz der Kosten berechen 
        # um uncertainty penalty hinzuzufügen
        mean_step_costs = per_step_costs
        var_step_costs = jnp.zeros_like(per_step_costs)

    var_discount = config.rollout_var_discount ** jnp.arange(config.horizon)
    var_penalty = config.rollout_var_cost * jnp.sum(
        var_step_costs * var_discount, axis=1
    )
    return jnp.sum(mean_step_costs, axis=1) + terminal_costs + var_penalty

#inkl sample cost mit cov in effizienter Tensorschreibweise
def _compute_noise_cost(
    noise: jax.Array,
    noise_sigma_inv: jax.Array,
    noise_abs_cost: bool,
) -> jax.Array:
    if noise_abs_cost:
        abs_noise = jnp.abs(noise)
        quad = jnp.einsum(
            "ktd,df,ktf->kt", abs_noise, jnp.abs(noise_sigma_inv), abs_noise
        )
    else:
        quad = jnp.einsum("ktd,df,ktf->kt", noise, noise_sigma_inv, noise)
    return 0.5 * jnp.sum(quad, axis=1)

def _compute_weights(costs: jax.Array, lambda_: float) -> jax.Array:
    min_cost = jnp.min(costs)
    scaled = -(costs - min_cost) / lambda_
    return jax.nn.softmax(scaled)
