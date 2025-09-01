import numpy as np
import matplotlib.pyplot as plt

from Quadrotor import Quadrotor
from Newtons_Method import gen_trajectory
import Simulator as sim
import Cost_Function as cost
import Ref_Trajectory as rf
import LQR as LQR

Drone = Quadrotor()

Qt1 = cost.Qt1
Qt2 = cost.Qt2
Rt1 = cost.Rt
Qf = cost.Pf
params = [Drone.M, Drone.m, Drone.J, Drone.g, Drone.L, Drone.k]

T = Drone.T
dt = Drone.dt
ns = Drone.ns
nu = Drone.nu

# Initial state
x = np.zeros((T, ns))
u = np.zeros((T, nu))
At = np.zeros((T, ns, ns))
Bt = np.zeros((T, ns, nu))
Qt = np.zeros((T, ns, ns))
Rt = np.zeros((T, nu, nu))



######################################################################
################## Set the Initial Perturbed State ###################
######################################################################

# perturbed_init_state = np.array([0.0, 0.1, 0.2,-0.2, 0.01, 0.0, 0., 0.])
# perturbed_init_state = np.array([0.01, 0.01, -0.1, 0.01, 0.01, 0, 0, 0]) 
perturbed_init_state = np.array([-0.1, 0.01, 0.01, 0.1, 0, 0.1, 0.01, 0])
# perturbed_init_state = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, -0.1])


######################################################################
############# Part I : Generate an Optimal Trajectory ################
######################################################################
u_d, x_d = gen_trajectory(trajectory_type = "Smooth", loop_type = "closed")[0:2]


######################################################################
######## Part II : Linearization Around Optimal Trajectory ###########
######################################################################

for t in range(T):
    At[t, :, :], Bt[t, :, :] = Drone.Jacobian(x_d[t, :], u_d[t, :])
    if t < T/2:
        Qt[t, :, :] = Qt1 #weighting matrix for the state
    
    else:
        Qt[t, :, :] = Qt2 
    
    Rt[t, :, :] = Rt1 #weighting matrix for control inputs

######################################################################
############### Part III : Computation of LQR Gains ##################
######################################################################
K, P = LQR.lti_LQR(At, Bt, Qt, Rt, Qf, T)



######################################################################
########### Part IV : Plot Perturbed Optimal Trajectory ##############
######################################################################

x[0, :] = perturbed_init_state  # Set the initial perturbed state

for t in range(T-1):
    u[t, :] = u_d[t, :]
    x[t+1, :] = Drone.Quadrotor_Dyanmics(x[t, :], u[t, :])

fig, axs = plt.subplots(10, 1, figsize=(10, 20), sharex=True)
fig.suptitle('Optimal vs Actual Without LQR', fontsize=16)
# Plot each state in a separate subplot and compare it with the refrence frame
for i in range(8):
    axs[i].plot(x[:, i], label=f'Actual_State {i + 1}')
    axs[i].plot(x_d[:, i],'--r',label=f'Desired_State {i + 1}')
    axs[i].set_ylabel(f'State {i + 1}')
    axs[i].grid()
for i in range(2):
    axs[i+8].plot(u[:, i], label=f'Actual_Input {i + 1}')
    axs[i+8].plot(u_d[:, i],'--r',label=f'Desired_Input {i + 1}')
    axs[i+8].set_ylabel(f'Input {i + 1}')
    axs[i+8].grid()
axs[9].set_xlabel('Time')
for ax in axs:
    ax.legend()
    ax.legend(loc = "upper left")
plt.show()


# Create a 5x2 subplot for states and inputs
fig, axs = plt.subplots(4, 2, figsize=(12, 20), sharex=True)
fig.suptitle('Optimal vs Actual Without LQR', fontsize=16)

# Plot each state in the first 8 subplots
for i in range(8):
    row, col = divmod(i, 2)
    axs[row, col].plot(x[:, i], label=f'Actual_State {i + 1}')
    axs[row, col].plot(x_d[:, i], '--r', label=f'Desired_State {i + 1}')
    axs[row, col].set_ylabel(f'State {i + 1}')
    axs[row, col].grid()
    axs[row, col].legend(loc="lower left")

# # Plot each input in the last 2 subplots
# for i in range(2):
#     axs[4, i].plot(u[:, i], label=f'Actual_Input {i + 1}')
#     axs[4, i].plot(u_d[:, i], '--r', label=f'Desired_Input {i + 1}')
#     axs[4, i].set_ylabel(f'Input {i + 1}')
#     axs[4, i].grid()
#     axs[4, i].legend(loc="lower left")

axs[3, 0].set_xlabel('Time')
axs[3, 1].set_xlabel('Time')
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()

######################################################################
######## Part V : Animation of Perturbed Optimal Trajectory ##########
######################################################################
sim.animate_quadrotor(x, u, dt, params)



######################################################################
########### Part VI : Plot Perturbed Optimal Trajectory ##############
##################### Feedback Control via LQR #######################
######################################################################

x[0, :] = perturbed_init_state  # Reset the initial perturbed state

for t in range(T-1):
    # Compute the control input using LQR feedback
    u[t, :] = u_d[t, :] + K[t, :] @ (x[t, :] - x_d[t, :]).T
    x[t+1, :] = Drone.Quadrotor_Dyanmics(x[t, :], u[t, :])

fig, axs = plt.subplots(10, 1, figsize=(10, 20), sharex=True)
fig.suptitle('Optimal Vs Actual With LQR', fontsize=16)

# Plot each state in a separate subplot and compare it with the refrence frame
for i in range(8):
    axs[i].plot(x[:, i], label=f'Actual_State {i + 1}')
    axs[i].plot(x_d[:, i],'--r',label=f'Desired_State {i + 1}')
    axs[i].set_ylabel(f'State {i + 1}')
    axs[i].grid()
for i in range(2):
    axs[i+8].plot(u[:, i], label=f'Control_LQR_Input {i + 1}')
    axs[i+8].plot(u_d[:, i],'--r',label=f'Desired_Control_Input {i + 1}')
    axs[i+8].set_ylabel(f'Input {i + 1}')
    axs[i+8].grid()
axs[9].set_xlabel('Time')
for ax in axs:
    ax.legend()
    ax.legend(loc = "upper right")
plt.show()

# Create a 5x2 subplot for states and inputs
fig, axs = plt.subplots(4, 2, figsize=(10, 20), sharex=True)
# fig, axs = plt.subplots(5, 2, figsize=(12, 20), sharex=True)
fig.suptitle('Optimal vs Actual With LQR', fontsize=16)

# Plot each state in the first 8 subplots
for i in range(8):
    row, col = divmod(i, 2)
    axs[row, col].plot(x[:, i], label=f'Actual_State {i + 1}')
    axs[row, col].plot(x_d[:, i], '--r', label=f'Desired_State {i + 1}')
    axs[row, col].set_ylabel(f'State {i + 1}')
    axs[row, col].grid()
    axs[row, col].legend(loc="lower right")

# Plot each input in the last 2 subplots
# for i in range(2):
#     axs[4, i].plot(u[:, i], label=f'Control_LQR_Input {i + 1}')
#     axs[4, i].plot(u_d[:, i], '--r', label=f'Desired_Control_Input {i + 1}')
#     axs[4, i].set_ylabel(f'Input {i + 1}')
#     axs[4, i].grid()
#     axs[4, i].legend(loc="lower right")

axs[3, 0].set_xlabel('Time')
axs[3, 1].set_xlabel('Time')
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()




######################################################################
################# Part VII : Plot of Tracking Error ##################
######################################################################

tracking_error = x - x_d
abs_tracking_error = np.abs(tracking_error)

bound1 = 0.05 * np.ones(len(abs_tracking_error))
bound2 = -bound1

# Plot the tracking error for each state
fig, axs = plt.subplots(4, 2, figsize=(10, 20), sharex=True)
plt.subplots_adjust(hspace=0.4, wspace=0.4)
fig.suptitle('Tracking Error for Each State', fontsize=16)

num_states = 8  # Assuming there are 8 states

i = 0
j = 0
k = 1
l = 0
while i < 4:
    while j < 2:
        axs[i, j].plot(tracking_error[:, l], '--b', label=f'Tracking Error State {l + 1}')
        axs[i, j].plot(bound1, '--r', label='Error bounds')
        axs[i, j].plot(bound2, '--r')
        axs[i, j].grid()
        axs[i, j].legend()
         # Set specific limits for the smaller subplot
        axs[i, j].set_ylim([-0.4, 0.4])
        j = j + 1
        l = l + 1
        pass
    i = i + 1
    k = k + 1
    j = 0
    pass

for ax in axs.flat:
    ax.set(xlabel='$Time$', ylabel='$Tracking Error$')
    pass

for ax in axs.flat:
    ax.label_outer()
    pass

plt.show()


# Plot the tracking error in logarithmic scale
fig, axs = plt.subplots(4, 2, figsize=(10, 20), sharex=True)
fig.suptitle('Tracking Error for Each State', fontsize=16, y = 0.95)
i = 0
j = 0
k = 1
l = 0
while i < 4:
    while j < 2:
        axs[i,j].plot(abs_tracking_error[:, l],)
        axs[i,j].set_yscale('log')
        axs[i,j].set_title(f'Error State {i + j + k}', fontsize=12)
        axs[i,j].grid()
        j = j + 1
        l = l + 1
        pass
    i = i + 1
    k = k + 1
    j = 0
    pass

for ax in axs.flat:
    ax.set(xlabel='$Time$', ylabel='$|xx - xx_des|$')
    pass

for ax in axs.flat:
    ax.label_outer()
    pass
    
plt.show()

######################################################################
######### Part V : Animation of Trajectory Tracking via LQR ##########
######################################################################
sim.animate_quadrotor(x, u, dt, params)