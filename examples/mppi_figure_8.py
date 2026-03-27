import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from crazyflow.control.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.physics import Physics
from crazyflow.utils import enable_cache

from submodules.Controllers_mppi import mppi_controller as mppi


DT = 0.1
HORIZON = 20


def figure8_reference(t: float):
    px = jnp.sin(t)
    py = 0.0
    pz = 0.5 * jnp.sin(2.0 * t) + 1.0

    vx = jnp.cos(t)
    vy = 0.0
    vz = jnp.cos(2.0 * t)

    pos_ref = jnp.array([px, py, pz])
    vel_ref = jnp.array([vx, vy, vz])
    return pos_ref, vel_ref


def get_mppi_state_from_sim(sim: Sim):
    pos = sim.data.states.pos[0, 0]
    vel = sim.data.states.vel[0, 0]
    return jnp.concatenate([pos, vel])


def dynamics(state, action, t):
    pos = state[:3]
    vel = state[3:]

    dpos = action[:3]
    dvel = action[3:]

    next_pos = pos + DT * vel + dpos
    next_vel = vel + dvel

    return jnp.concatenate([next_pos, next_vel])


def make_running_cost(global_step):
    def running_cost(state, action, t):
        pos = state[:3]
        vel = state[3:]

        t_ref = (global_step + t) * DT
        pos_ref, vel_ref = figure8_reference(t_ref)

        pos_error = pos - pos_ref
        vel_error = vel - vel_ref

        pos_cost = 80.0 * jnp.sum(pos_error ** 2)
        y_cost = 50.0 * pos_error[1] ** 2
        vel_cost = 10.0 * jnp.sum(vel_error ** 2)
        act_cost = 0.1 * jnp.sum(action ** 2)

        return pos_cost + y_cost + vel_cost + act_cost

    return running_cost


def make_terminal_cost(global_step):
    def terminal_cost(state, action):
        pos = state[:3]
        vel = state[3:]

        t_ref = (global_step + HORIZON - 1) * DT
        pos_ref, vel_ref = figure8_reference(t_ref)

        pos_error = pos - pos_ref
        vel_error = vel - vel_ref

        pos_terminal = 120.0 * jnp.sum(pos_error ** 2)
        y_terminal = 80.0 * pos_error[1] ** 2
        vel_terminal = 20.0 * jnp.sum(vel_error ** 2)

        return pos_terminal + y_terminal + vel_terminal

    return terminal_cost


def build_mppi():
    config, state = mppi.create(
        nx=6,
        nu=6,
        noise_sigma=jnp.eye(6) * 0.05,
        num_samples=800,
        horizon=HORIZON,
        lambda_=1.0,
        u_min=jnp.array([-0.05, -0.05, -0.05, -0.1, -0.1, -0.1]),
        u_max=jnp.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1]),
        step_dependent_dynamics=True,
    )
    return config, state


def build_state_command(global_step, mppi_action):
    dpos = mppi_action[:3]
    dvel = mppi_action[3:]

    t_ref = global_step * DT
    pos_ref, vel_ref = figure8_reference(t_ref)

    pos_cmd = pos_ref + dpos
    vel_cmd = vel_ref + dvel

    cmd = jnp.zeros(13)
    cmd = cmd.at[0:3].set(pos_cmd)         # x, y, z
    cmd = cmd.at[3:6].set(vel_cmd)         # vx, vy, vz
    cmd = cmd.at[6:9].set(jnp.zeros(3))    # ax, ay, az
    cmd = cmd.at[9].set(0.0)               # yaw
    cmd = cmd.at[10:13].set(jnp.zeros(3))  # roll_rate, pitch_rate, yaw_rate

    return cmd


def main():
    print("Start main")
    enable_cache()

    sim = Sim(
        n_worlds=1,
        n_drones=1,
        control=Control.state,
        physics=Physics.so_rpy,
        device="cpu",
        freq=500,
        state_freq=100,
        attitude_freq=500,
        force_torque_freq=500,
    )
    print("Sim created")

    sim.reset()
    print("Sim reset done")

    config, mppi_state = build_mppi()
    print("MPPI created")

    states = []
    cmds = []
    ref_positions = []

    print("Before first step")
    for k in range(200):
        current_obs = get_mppi_state_from_sim(sim)

        running_cost_k = make_running_cost(k)
        terminal_cost_k = make_terminal_cost(k)

        action, mppi_state = mppi.command(
            config,
            mppi_state,
            current_obs,
            dynamics=dynamics,
            running_cost=running_cost_k,
            terminal_cost=terminal_cost_k,
        )

        state_cmd = build_state_command(k, action)

        # shape: (n_worlds, n_drones, 13)
        sim.state_control(state_cmd[None, None, :])

        # mehrere Substeps pro MPC-Schritt
        sim.step(n_steps=10)

        try:
            sim.render()
        except Exception:
            pass

        states.append(np.asarray(current_obs))
        cmds.append(np.asarray(state_cmd))
        ref_positions.append(np.asarray(figure8_reference(k * DT)[0]))

    states = np.stack(states)
    cmds = np.stack(cmds)
    ref_positions = np.stack(ref_positions)

    print("Final state:", states[-1])
    print("Final position:", states[-1, :3])

    print("Trajectory shape:", states.shape)
    print("First 5 trajectory points:")
    print(states[:5, :3])
    print("Last 5 trajectory points:")
    print(states[-5:, :3])

    Path("figures").mkdir(parents=True, exist_ok=True)

    trajectory = np.column_stack([
        np.arange(states.shape[0]) * DT,
        states[:, 0], states[:, 1], states[:, 2],
        states[:, 3], states[:, 4], states[:, 5],
        ref_positions[:, 0], ref_positions[:, 1], ref_positions[:, 2],
    ])

    header = "time,px,py,pz,vx,vy,vz,ref_x,ref_y,ref_z"
    np.savetxt(
        "figures/mppi_figure8_trajectory.csv",
        trajectory,
        delimiter=",",
        header=header,
        comments="",
    )
    print("Trajectory saved to figures/mppi_figure8_trajectory.csv")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        ref_positions[:, 0],
        ref_positions[:, 1],
        ref_positions[:, 2],
        linestyle="--",
        label="reference",
    )
    ax.plot(
        states[:, 0],
        states[:, 1],
        states[:, 2],
        marker="o",
        markersize=2,
        label="mppi",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("MPPI Figure-8 Tracking with Sim.state_control")
    ax.legend()

    plt.savefig("figures/mppi_figure8_sim_state_control.png", dpi=200, bbox_inches="tight")
    print("Plot saved to figures/mppi_figure8_sim_state_control.png")

    try:
        sim.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()