
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


# =========================
# Reference trajectory
# =========================
def figure8_reference(t: jnp.ndarray, A: float = 2.0, B: float = 1.0, w: float = 0.6):
    """
    Figure-8 trajectory in 2D.
    Returns position and velocity references for time t.
    """
    x = A * jnp.sin(w * t)
    y = B * jnp.sin(2.0 * w * t)

    vx = A * w * jnp.cos(w * t)
    vy = 2.0 * B * w * jnp.cos(2.0 * w * t)

    return jnp.array([x, y]), jnp.array([vx, vy])


# =========================
# Point-mass dynamics
# State: [x, y, vx, vy]
# Input: [ax, ay]
# =========================
def dynamics(x: jnp.ndarray, u: jnp.ndarray, dt: float):
    px, py, vx, vy = x
    ax, ay = u

    px_next = px + vx * dt
    py_next = py + vy * dt
    vx_next = vx + ax * dt
    vy_next = vy + ay * dt

    return jnp.array([px_next, py_next, vx_next, vy_next])


# =========================
# Cost function
# =========================
def stage_cost(
    x: jnp.ndarray,
    u: jnp.ndarray,
    t: float,
    Q_pos: float,
    Q_vel: float,
    R_u: float,
):
    pos_ref, vel_ref = figure8_reference(t)

    pos = x[:2]
    vel = x[2:]

    pos_err = pos - pos_ref
    vel_err = vel - vel_ref

    cost_pos = Q_pos * jnp.sum(pos_err**2)
    cost_vel = Q_vel * jnp.sum(vel_err**2)
    cost_u = R_u * jnp.sum(u**2)

    return cost_pos + cost_vel + cost_u


def terminal_cost(x: jnp.ndarray, t: float, Qf_pos: float, Qf_vel: float):
    pos_ref, vel_ref = figure8_reference(t)
    pos = x[:2]
    vel = x[2:]

    pos_err = pos - pos_ref
    vel_err = vel - vel_ref

    return Qf_pos * jnp.sum(pos_err**2) + Qf_vel * jnp.sum(vel_err**2)


# =========================
# Rollout over one sampled control sequence
# =========================
def rollout_single(
    x0: jnp.ndarray,
    U_nominal: jnp.ndarray,
    noise: jnp.ndarray,
    t0: float,
    dt: float,
    u_min: float,
    u_max: float,
    Q_pos: float,
    Q_vel: float,
    R_u: float,
    Qf_pos: float,
    Qf_vel: float,
):
    """
    x0: shape (4,)
    U_nominal: shape (H, 2)
    noise: shape (H, 2)
    """
    H = U_nominal.shape[0]

    def step_fn(carry, k):
        x, cost = carry
        u = jnp.clip(U_nominal[k] + noise[k], u_min, u_max)
        t = t0 + k * dt
        cost = cost + stage_cost(x, u, t, Q_pos, Q_vel, R_u)
        x_next = dynamics(x, u, dt)
        return (x_next, cost), x_next

    (x_final, total_cost), xs = jax.lax.scan(
        step_fn,
        (x0, 0.0),
        jnp.arange(H),
    )

    total_cost = total_cost + terminal_cost(x_final, t0 + H * dt, Qf_pos, Qf_vel)
    return total_cost, xs


# Vectorize rollouts over many samples
rollout_batch = jax.jit(
    jax.vmap(
        rollout_single,
        in_axes=(None, None, 0, None, None, None, None, None, None, None, None, None),
    )
)


# =========================
# MPPI controller
# =========================
def mppi_step(
    key: jax.Array,
    x0: jnp.ndarray,
    U_nominal: jnp.ndarray,
    t0: float,
    dt: float,
    num_samples: int,
    noise_sigma: float,
    lam: float,
    u_min: float,
    u_max: float,
    Q_pos: float,
    Q_vel: float,
    R_u: float,
    Qf_pos: float,
    Qf_vel: float,
):
    """
    One MPPI update step.
    """
    H, nu = U_nominal.shape

    noise = noise_sigma * jax.random.normal(key, shape=(num_samples, H, nu))

    costs, _ = rollout_batch(
        x0,
        U_nominal,
        noise,
        t0,
        dt,
        u_min,
        u_max,
        Q_pos,
        Q_vel,
        R_u,
        Qf_pos,
        Qf_vel,
    )

    rho = jnp.min(costs)
    weights_unnorm = jnp.exp(-(costs - rho) / lam)
    weights = weights_unnorm / (jnp.sum(weights_unnorm) + 1e-8)

    weighted_noise = jnp.sum(weights[:, None, None] * noise, axis=0)
    U_new = jnp.clip(U_nominal + weighted_noise, u_min, u_max)

    return U_new, costs, weights


# =========================
# Closed-loop simulation
# =========================
def simulate():
    # Time / horizon
    dt = 0.05
    horizon = 30
    sim_steps = 300

    # MPPI params
    num_samples = 512
    noise_sigma = 0.35
    lam = 1.0

    # Control bounds
    u_min = -3.0
    u_max = 3.0

    # Cost weights
    Q_pos = 20.0
    Q_vel = 2.0
    R_u = 0.05
    Qf_pos = 40.0
    Qf_vel = 4.0

    # Initial state
    x = jnp.array([0.0, -1.5, 0.0, 0.0])

    # Initial nominal sequence
    U_nominal = jnp.zeros((horizon, 2))

    xs = [x]
    us = []
    refs = []

    key = jax.random.PRNGKey(0)

    for k in range(sim_steps):
        t = k * dt
        key, subkey = jax.random.split(key)

        U_nominal, costs, weights = mppi_step(
            subkey,
            x,
            U_nominal,
            t,
            dt,
            num_samples,
            noise_sigma,
            lam,
            u_min,
            u_max,
            Q_pos,
            Q_vel,
            R_u,
            Qf_pos,
            Qf_vel,
        )

        # Apply first control (receding horizon)
        u = U_nominal[0]
        x = dynamics(x, u, dt)

        # Shift control sequence
        U_nominal = jnp.concatenate([U_nominal[1:], U_nominal[-1:]], axis=0)

        pos_ref, _ = figure8_reference(t)

        xs.append(x)
        us.append(u)
        refs.append(pos_ref)

    xs = jnp.stack(xs)
    us = jnp.stack(us)
    refs = jnp.stack(refs)

    return xs, us, refs, dt


# =========================
# Plotting
# =========================
def plot_results(xs, us, refs, dt):
    t_state = jnp.arange(xs.shape[0]) * dt
    t_ctrl = jnp.arange(us.shape[0]) * dt

    plt.figure(figsize=(8, 6))
    plt.plot(refs[:, 0], refs[:, 1], "--", label="reference")
    plt.plot(xs[:, 0], xs[:, 1], label="trajectory")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("MPPI Figure-8 Tracking (2D Point Mass)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(t_state, xs[:, 0], label="x")
    plt.plot(t_state, xs[:, 1], label="y")
    plt.xlabel("time [s]")
    plt.ylabel("position")
    plt.title("Position states")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(t_ctrl, us[:, 0], label="ax")
    plt.plot(t_ctrl, us[:, 1], label="ay")
    plt.xlabel("time [s]")
    plt.ylabel("control")
    plt.title("Control inputs")
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    xs, us, refs, dt = simulate()
    plot_results(xs, us, refs, dt)


if __name__ == "__main__":
    main()