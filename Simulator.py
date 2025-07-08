import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Quadrotor import Quadrotor
from Ref_Trajectory import step_ref_trajectory


Drone = Quadrotor()

params = [Drone.M, Drone.m, Drone.J, Drone.g, Drone.L, Drone.k]



def simulate_quadrotor_with_history(initial_state, control_inputs, dt, params):
    num_steps = len(control_inputs)
    states = np.zeros((num_steps, len(initial_state)))
    states[0, :] = initial_state

    for i in range(1, num_steps):
        states[i, :] = Drone.Quadrotor_Dyanmics(states[i-1, :], control_inputs[i-1], dt, params)
    return states




def animate_quadrotor(simulated_states, control_inputs, dt, params):
    fig, ax = plt.subplots()
    ax.set_xlim(-1.5, 3)
    ax.set_ylim(-1.5, 3)

    quadrotor_right, = ax.plot([], [], 'bo-', markersize=9)
    quadrotor_left, = ax.plot([], [], 'bo-', markersize=9)
    quadrotor_line, = ax.plot([], [], 'b-', linewidth=2)
    load, = ax.plot([], [], 'ro-', markersize=4)
    line_between_load_and_axes, = ax.plot([], [], 'g-', linewidth=1)

    # Initialize footprint
    footprint_line, = ax.plot([], [], 'k-', linewidth=0.5)

    def init():
        quadrotor_right.set_data([], [])
        quadrotor_left.set_data([], [])
        quadrotor_line.set_data([], [])
        load.set_data([], [])
        line_between_load_and_axes.set_data([], [])
        footprint_line.set_data([], [])
        return quadrotor_right, quadrotor_left, quadrotor_line, load, line_between_load_and_axes, footprint_line

    def update(frame):
        xp, yp, alpha, theta, _, _, _, _ = simulated_states[frame, :]

        quadrotor_right.set_data([xp + 0.1 * np.cos(theta)], [yp + 0.1 * np.sin(theta)])
        quadrotor_left.set_data([xp - 0.1 * np.cos(theta)], [yp - 0.1 * np.sin(theta)])
        quadrotor_line.set_data([xp + 0.1 * np.cos(theta), xp - 0.1 * np.cos(theta)],
                                [yp + 0.1 * np.sin(theta), yp - 0.1 * np.sin(theta)])

        load_x = xp + Drone.L * np.sin(alpha)
        load_y = yp - Drone.L * np.cos(alpha)

        load.set_data([xp, load_x], [yp, load_y])
        line_between_load_and_axes.set_data([xp, (xp + load_x) / 2], [yp, (yp + load_y) / 2])

        footprint_history.append((xp, yp))
        footprint_line.set_data(*zip(*footprint_history))

        return quadrotor_right, quadrotor_left, quadrotor_line, load, line_between_load_and_axes, footprint_line

    footprint_history = []  # Initialize an empty list to store footprint history
    animation = FuncAnimation(fig, update, frames=len(control_inputs), init_func=init, blit=True, interval=1)

    plt.show()