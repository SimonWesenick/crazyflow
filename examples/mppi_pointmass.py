
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

    pos_terminal = 50.0 * jnp.sum((pos-GOAL)**2)
    vel_terminal = 20.0 *jnp.sum(vel**2)

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
        num_samples=500,
        horizon=25,
        lambda_=1,
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

    plt.figure(figsize=(6, 6))
    plt.plot(states[:, 0], states[:, 1], marker="o", markersize=3)
    plt.scatter(GOAL[0], GOAL[1], marker="x", s=100, label="goal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("MPPI Point Mass Trajectory")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)

    Path("figures").mkdir(parents=True, exist_ok=True)
    plt.savefig("figures/mppi_pointmass_trajectory.png", dpi=200, bbox_inches="tight")
    print("Plot saved to figures/mppi_pointmass_trajectory.png")


if __name__ == "__main__":
    main()