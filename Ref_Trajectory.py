import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline as Spline
from Quadrotor import Quadrotor



Drone = Quadrotor()
Kv = 1 # Velocity gain for the quadrotor

# Equilibrium States
# Equilibrium condition of the quadrotor at starting point
#[x_pos1, y_pos2, alpha1, theta1, vel_x1, vel_y1, angvel_alpha1, angvel_theta1]
state_eq1 = np.array([0,0,0,0,0,0,0,0])


# Equilibrium condition of the quadrotor at final point
#[x_pos2, y_pos2, alpha2, theta2, vel_x2, vel_y2, angvel_alpha2, angvel_theta2]
state_eq2 = np.array([1,1,0,0,0,0,0,0])
# state_eq2 = np.array([1, 1, np.pi, 0, 0, 0, 0, 0])  # fancy trajectroy for LQR tracking " stablising the pendulum in the up-right direction

# Force rquired to hover at stable position Fs = (M+m)*g, Fd = 0
Fr = (Drone.M + Drone.m)* Drone.g / 2 # Thrust force by right propeller 
Fl = (Drone.M + Drone.m)* Drone.g / 2 # Thrust force by left propeller



def step_ref_trajectory(T):
    """
    Generates a step reference trajectory for the quadrotor.
    
    Parameters:
    T: float - time in seconds
    
    Returns:
    np.array - Reference trajectory, Reference Inputs
    """
    # Time points for the trajectory
    t_span = np.array([0.0, 0.5, 0.5, 1.0])
    # t_span = np.array([0.0, 0.25, 0.75, 1.0])

    # Linear interpolation for positions
    pos = np.array([state_eq1, state_eq1, state_eq2, state_eq2])
    time_points = np.linspace(0.0, 1, T)
    ref_trajectory = np.zeros((len(time_points), len(state_eq1)))

    u_ref = np.zeros((len(time_points), 2))  # Control inputs for the reference trajectory

    for i in range(len(pos[0])):
        # Interpolate position for each state variable
        ref_trajectory[:, i] = np.interp(time_points, t_span, pos[:, i])

    # Contant Velocity
    const_vel = 0 # Set your desired constant velocity
    # const_vel = 0.5

    for i in range(T):
        if 0 < ref_trajectory[i, 0] < 1:
            ref_trajectory[i, 4] = const_vel
            # The idea is: the farther you are from the goal, the faster you should move
            # ref_trajectory[i, 4] = Kv * np.sqrt ((ref_trajectory[i, 0] - state_eq2[0])**2 + (ref_trajectory[i, 1] - state_eq2[1])**2)
        if 0 < ref_trajectory[i, 1] < 1:
            ref_trajectory[i, 5] = const_vel
            # ref_trajectory[i, 5] = Kv * np.sqrt ((ref_trajectory[i, 0] - state_eq2[0])**2 + (ref_trajectory[i, 1] - state_eq2[1])**2)

            u_ref[1, :] = [Fr + Fl, Fr - Fl]
        else:
            u_ref[1, :] = [Fr + Fl, Fr - Fl]
    
    # Plot all states in subplots
    
    fig, axs = plt.subplots(10, 1, figsize=(10, 20), sharex=True,dpi=100)
    fig.suptitle('Reference Curve', fontsize=20,y = 0.95)
    # Plot each state in a separate subplot
    for i in range(8):
        axs[i].plot(time_points, ref_trajectory[:, i],'--r', label=f'State {i + 1}')
        axs[i].set_ylabel(f'State {i + 1}')
        axs[i].grid()
    for i in range(2):
        axs[8+i].plot(time_points, u_ref[:, i],'--r', label=f'input {i + 1}')
        axs[8+i].set_ylabel(f'input {i + 1}')
        axs[8+i].grid()
    # Add labels and legends
    axs[9].set_xlabel('Time')
    for ax in axs:
        ax.legend()
    
    plt.show()


    ##################
    # Plot each states 
    ##################
        # for i in range(8):
        #     plt.figure(f'Reference State {i + 1}')
        #     plt.title(f'Reference State {i + 1}', fontsize=16)
        #     plt.plot(time_points, ref_trajectory[:, i],'--r', label=f'State {i + 1}')
        #     plt.grid()

        # for i in range(2):
        #     plt.figure(f'Refrence Input {i + 1}')
        #     plt.title(f'Refrence Input {i + 1}', fontsize=16)  
        #     plt.plot(time_points, u_ref[:, i],'--r', label=f'input {i + 1}')
        #     plt.grid()

        # plt.show()



    return ref_trajectory, u_ref



def smooth_ref_trajectory(T):
    """
    Generates a smooth reference trajectory for the quadrotor.
    
    Parameters:
    T: int - Number of time steps
    
    Returns:
    np.array - Smoothed reference trajectory
    """
    # Calculate the tangent vector at the end points
    m1 = 0 # Slope at the start point
    m2 = 0 # Slope at the end point
    T_eq = int(T / 3) # The middle third of the trajectory will be the smooth transition.

    # Create a cubic Hermite spine interpolation

    time_points = np.linspace(0.0, 1, T)
    ref_trajectory = np.zeros((len(time_points), len(state_eq1)))

    for i in range(len(state_eq1)):
        if state_eq1[i] == state_eq2[i]:
            continue
        else:
            spline = Spline([state_eq1[i], state_eq2[i]], [state_eq1[i], state_eq2[i]], [m1, m2])

            #Generating points for plotting the spline
            xx = np.linspace(state_eq1[i], state_eq2[i], T - T_eq)
            ref_trajectory[T_eq : T-T_eq, i] = spline(xx)
            ref_trajectory[T - T_eq:, i] = state_eq2[i]

        # Linear interpolation for the first and last third of the trajectory
        u_ref = np.zeros((len(time_points), 2))  # Control inputs for the reference trajectory

        #Constant Velocity
        const_vel = 0 # Set your desired constant velocity
        for i in range(T):
            if 0.1 < ref_trajectory[i, 0] < 0.9:
                ref_trajectory[i, 4] = const_vel
                # The idea is: the farther you are from the goal, the faster you should move
                # ref_trajectory[i, 4] = Kv * np.sqrt ((ref_trajectory[i, 0] - ref_trajectory[-1, 0])**2 + (ref_trajectory[i, 1] - ref_trajectory[-1, 1])**2)
            
            if 0.1 < ref_trajectory[i, 1] < 0.9:
                ref_trajectory[i, 5] = const_vel
                # ref_trajectory[i, 4] = Kv * np.sqrt ((ref_trajectory[i, 0] - ref_trajectory[-1, 0])**2 + (ref_trajectory[i, 1] - ref_trajectory[-1, 1])**2)
                u_ref[i, :] = [Fr + Fl, 0]

            else:
                u_ref[i, :] = [Fr + Fl, 0]


    # Plot all states in subplots
        fig, axs = plt.subplots(len(state_eq1) + 2, 1, figsize=(10, 15), sharex=True,dpi=100)
    fig.suptitle('Reference Curve', fontsize=16, y = 0.95)
    # Plot each state in a separate subplot
    for i in range(len(state_eq1)):
        axs[i].plot(time_points, ref_trajectory[:, i],'--r', label=f'State {i + 1}')
        axs[i].set_ylabel(f'State {i + 1}')
        axs[i].grid()
    # Plot control inputs in separate subplots
    axs[len(state_eq1)].plot(time_points, u_ref[:, 0],'--r', label='Control Input 1')
    axs[len(state_eq1)].set_ylabel('Control Input 1')
    axs[len(state_eq1)].grid()
    axs[len(state_eq1) + 1].plot(time_points, u_ref[:, 1],'--r', label='Control Input 2')
    axs[len(state_eq1) + 1].set_ylabel('Control Input 2')
    axs[len(state_eq1)+1].grid()
    # Add labels and legends
    axs[len(state_eq1) + 1].set_xlabel('Time')
    for ax in axs:
        ax.legend()
    plt.show()

    

##################
# Plot each states 
##################
    # for i in range(8):
    #     plt.figure(f'Reference State {i + 1}')
    #     plt.title(f'Reference State {i + 1}', fontsize=16)
    #     plt.plot(time_points, trajectory_points[:, i],'--r', label=f'State {i + 1}')
    #     plt.grid()
 
    # for i in range(2):
    #     plt.figure(f'Refrence Input {i + 1}')
    #     plt.title(f'Refrence Input {i + 1}', fontsize=16)
    #     plt.plot(time_points, uu_ref[:, i],'--r', label=f'input {i + 1}')
    #     plt.grid()

    # plt.show()




    return ref_trajectory, u_ref