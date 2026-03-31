
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from submodules.Controllers_mppi import mppi_controller as mppi
from pathlib import Path


DT = 0.1
GOAL = jnp.array([2.0, 2.0])


def dynamics(state, action):
    px, py, vx, vy = state
    ax, ay = action

    next_state = jnp.array([
        px + DT * vx,
        py + DT * vy,
        vx + DT * ax,
        vy + DT * ay,
    ])
    return next_state

def terimnal_cost(state, action):
    px,py,vx,vy = state
    pos = jnp.array([px, py])
    vel = jnp.array([vx, vy])

    pos_terminal = 70 * jnp.sum((pos-GOAL)**2)
    vel_terminal = 30 *jnp.sum(vel**2)

    return pos_terminal + vel_terminal
def running_cost(state, action):
    px, py, vx, vy = state
    pos = jnp.array([px, py])
    vel = jnp.array([vx, vy])

    pos_cost = 10.0 * jnp.sum((pos - GOAL) ** 2)
    vel_cost = 10* jnp.sum(vel ** 2)
    act_cost = 0.2 * jnp.sum(action ** 2)

    return pos_cost + vel_cost + act_cost


def main():
    config, mppi_state = mppi.create(
        nx=4,
        nu=2,
        noise_sigma=jnp.eye(2) * 0.3,
        num_samples=2000,
        horizon=25,
        lambda_=0.9,
        u_min=jnp.array([-1.0, -1.0]),
        u_max=jnp.array([1.0, 1.0]),
    )

    current_obs = jnp.array([0.0, 0.0, 0.0, 0.0])

    #jax wiederholte aufrufe just-in-time
    jitted_command = jax.jit(
    mppi.command,
    static_argnames=["config", "dynamics", "running_cost", "terminal_cost"],
)

    states = [current_obs]
    actions = []

    for _ in range(80):
        action, mppi_state = jitted_command(
            config,
            mppi_state,
            current_obs,
            dynamics=dynamics,
            running_cost=running_cost,
            terminal_cost=terimnal_cost,
        )

        current_obs = dynamics(current_obs, action)

        states.append(current_obs)
        actions.append(action)

    states = jnp.stack(states)
    actions = jnp.stack(actions)

    print("Final state:", states[-1])
    print("Final position:", states[-1, :2])

    time = jnp.arange(len(states)) * DT
    speed = jnp.sqrt(states[:, 2] ** 2 + states[:, 3] ** 2)

    fig, (ax_traj, ax_vel) = plt.subplots(1, 2, figsize=(12, 5))

    ax_traj.plot(states[:, 0], states[:, 1], marker="o", markersize=3, label="trajectory")
    ax_traj.scatter(GOAL[0], GOAL[1], marker="x", s=100, color="red", label="goal")
    ax_traj.set_xlabel("x")
    ax_traj.set_ylabel("y")
    ax_traj.set_title("MPPI Point Mass Trajectory")
    ax_traj.axis("equal")
    ax_traj.legend()
    ax_traj.grid(True)

    ax_vel.plot(time, states[:, 2], label="vx")
    ax_vel.plot(time, states[:, 3], label="vy")
    ax_vel.plot(time, speed, label="|v|", linestyle="--", color="black")
    ax_vel.axhline(0, color="gray", linewidth=0.8)
    ax_vel.set_xlabel("time [s]")
    ax_vel.set_ylabel("velocity [m/s]")
    ax_vel.set_title("Velocity over Time")
    ax_vel.legend()
    ax_vel.grid(True)

    fig.tight_layout()

    Path("figures").mkdir(parents=True, exist_ok=True)
    plt.savefig("figures/mppi_pointmass_trajectory.png", dpi=200, bbox_inches="tight")
    print("Plot saved to figures/mppi_pointmass_trajectory.png")


if __name__ == "__main__":
    main()