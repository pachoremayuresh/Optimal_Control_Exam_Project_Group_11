import numpy as np
import matplotlib.pyplot as plt


from Quadrotor import Quadrotor
from solver_MPC import solver_linear_mpc
import Simulator as sim
import Ref_Trajectory as rf
from Newtons_Method import gen_trajectory


Drone = Quadrotor()

T = Drone.T
dt = Drone.dt
ns = Drone.ns
nu = Drone.nu
params = [Drone.M, Drone.m, Drone.J, Drone.g, Drone.L, Drone.k]

At = np.zeros((T, ns, ns))
Bt = np.zeros((T, ns, nu))
Qt = np.zeros((T, ns, ns))
Rt = np.zeros((T, nu, nu))

Qt1 = 1*np.diag([1000,1000,10,100,10,10,0,1])
Qt2 = 1*np.diag([1000,1000,10,100,10,10,0,1])
Rt1 = 1*np.eye(nu)

Qf = Qt2

######################################################################
################## Set the Initial Perturbed State ###################
######################################################################


initiale_state_perturbed = np.array([-0.2, 0.1, 0.2,-0.2, 0.01, 0.01, 0.01, 0.01])
# initiale_state_perturbed = np.array([0.01, 0.01, -0.1, 0.01, 0.01, 0, 0, 0]) 
# initiale_state_perturbed = np.array([-0.1, 0.01, 0.01, 0.1, 0, 0.1, 0.01, 0])
# initiale_state_perturbed = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, -0.1])
perturbeation = np.array([-0.1, 0.1,0.1,0, 0.1, 0.0, 0.0, 0.0]) # wind disturbance

######################################################################
############# Part I : Generate an Optimal Trajectory ################
######################################################################
u_opt, x_opt, Pt = gen_trajectory(trajectory_type = "Smooth", loop_type = "closed")[0:3]


######################################################################
######## Part II : Linearization Around Optimal Trajectory ###########
######################################################################

for t in range(T):
    At[t, :, :], Bt[t, :, :] = Drone.Jacobian(x_opt[t, :], u_opt[t, :])
    if t < T/2:
        Qt[t, :, :] = Qt1 #weighting matrix for the state
    
    else:
        Qt[t, :, :] = Qt2 
    
    Rt[t, :, :] = Rt1 #weighting matrix for control inputs


######################################################################
############ Part III : Generate Real Optimal Trajectory #############
######################################################################

u_real_opt = np.zeros((T,nu)) # the real optimal open loop input will be affected by the perturbed initial state
x_real_opt = np.zeros((T,ns)) # real optimal trajectroy affected by the disturbances 
x_real_opt[0,:] = initiale_state_perturbed.squeeze() 

for t in range(T-1):
    u_real_opt[t,:] = u_opt[t,:]
    x_real_opt[t+1,:] = Drone.Quadrotor_Dyanmics(x_real_opt[t,:], u_real_opt[t,:])

    ###############################
    #Add external disturbance
    ###############################
    # if t == (T/ 2) :
    #   xx_real_opt[t+1,:] += perturbeation # wind disturbance


######################################################################
################ Part IV : Model Predictive Control ##################
######################################################################


#Parameters for MPC

T_pred = 5 # prediction horizon
T_sim = T - T_pred # simulation horizon

umax = 100
umin = -umax

#Constraints on the states and inputs
u_max_constraints = np.array([10, 10])
u_min_constraints = -u_max_constraints

x_max_constraints = np.array([2, 2, 2, 2, 10, 10, 10, 10])
x_min_constraints = -x_max_constraints

x_real_mpc = np.zeros((T, ns))
u_real_mpc = np.zeros((T, nu))

x_d = np.zeros((T_pred, ns))
u_d = np.zeros((T_pred, nu))

x_mpc = np.zeros((T, T_pred, ns))
u_mpc = np.zeros((T, T_pred, nu))

x_real_mpc[0, :] = initiale_state_perturbed.squeeze()

A = np.zeros((T_pred, ns, ns))
B = np.zeros((T_pred, ns, nu))

#Run the MPC
for t in range(T_sim):

    Q = Qt[t, :, :]
    R = Rt[t, :, :]

    # Compute linearized system dynamics (Jacobian matrices) at the current state and input
    A = At[t:t+T_pred, :, :]
    B = Bt[t:t+T_pred, :, :]

    # Desired trajectory over the prediction horizon
    x_d = x_opt[t:t+T_pred, :]
    u_d = u_opt[t:t+T_pred, :]

    # Get the initial state for the current MPC iteration
    x_t_mpc = x_real_mpc[t, :]

    if t%T_pred == 0:
        print(f"MPC iteration: {t}")
    
    # Solve the MPC optimization problem
    u_real_mpc[t, :], x_mpc[t, :, :], u_mpc[t, :, :] = solver_linear_mpc(x_d, u_d, A, B, Q, R, Pt[t,:,:], x_t_mpc,
                                                                        u_min_constraints, u_max_constraints,
                                                                        x_min_constraints, x_max_constraints, T_pred = T_pred)
    
    # Apply the first control input to the real system
    x_real_mpc[t+1, :] = Drone.Quadrotor_Dyanmics(x_real_mpc[t, :], u_real_mpc[t, :])

    ###############################
    #4.2.7 Add external disturbance
    ###############################
    # if  t == (T/ 2):
    #   xx_real_mpc[t+1,:] += perturbeation #wind disturbance


######################################################################
############ Part V : Plot Optimal Trajectory Vs Actual ##############
########################### Without MPC ##############################

fig, axs = plt.subplots(8, 1, figsize=(10, 20), sharex=True)
fig.suptitle('Optimal Vs Actual Without MPC', fontsize=16)
# Plot each state in a separate subplot and compare it with the refrence frame
for i in range(8):
    axs[i].plot(x_real_opt[:T_sim, i],'b', label=f'State {i + 1}')
    axs[i].plot(x_opt[:T_sim, i],'--r', label=f'State_ref {i + 1}')
    axs[i].set_ylabel(f'State {i + 1}')
    axs[i].grid()
axs[7].set_xlabel('Time')
for ax in axs:
    ax.legend()
plt.show()

# # show each state in separte figure
# for i in range(8):
#     plt.figure()
#     plt.plot(x_real_opt[:T_sim, i],'b', label=f'State {i + 1}')
#     plt.plot(x_opt[:T_sim, i],'--r', label=f'State_ref {i + 1}')
#     plt.ylabel(f'State {i + 1}')
#     plt.xlabel('Time')
#     plt.grid()
#     plt.legend()
# plt.show() 

######################################################################
############ Part VI : Plot Optimal Trajectory Vs Actual #############
############################ With MPC ################################
#######################################################################

fig, axs = plt.subplots(10, 1, figsize=(10, 20), sharex=True,dpi=100)
fig.suptitle('Optimal Vs Actual With MPC', fontsize=16)
# Plot each state in a separate subplot and compare it with the refrence curve
for i in range(8):
    axs[i].plot(x_real_mpc[:T_sim, i],'b',label=f'State {i + 1}')
    axs[i].plot(x_opt[:T_sim, i],'--r', label=f'State_ref {i + 1}')
    axs[i].set_ylabel(f'State {i + 1}')
    axs[i].grid()
for i in range(2):
    axs[8+i].plot(u_real_mpc[:, i],'b',label=f'mpc_input {i + 1}')
    axs[8+i].plot(u_opt[:, i],'--r', label=f'ref_input {i + 1}')
    axs[8+i].set_ylabel(f'input {i + 1}')
    axs[8+i].grid()
axs[9].set_xlabel('Time')
for ax in axs:
    ax.legend()
plt.show()

        
# # show each state in separte figure   
# for i in range(8):
#     plt.figure()
#     plt.plot(x_real_mpc[:T_sim, i],'b', label=f'State {i + 1}')
#     plt.plot(x_opt[:T_sim, i],'--r', label=f'State_ref {i + 1}')
#     plt.ylabel(f'State {i + 1}')
#     plt.xlabel('Time')
#     plt.grid()
#     plt.legend()
# plt.show()
# for i in range(2):
#     plt.figure()
#     plt.plot(u_real_mpc[:T_sim-1, i],'b', label=f'mpc_input {i + 1}')
#     plt.plot(u_opt[:T_sim-1, i],'--r', label=f'ref_input {i + 1}')
#     plt.ylabel(f' MPC Without Delta_u input {i + 1}')
#     plt.xlabel('Time')
#     plt.grid()
#     plt.legend()
# plt.show()

######################################################################
################# Part VII : Plot of Tracking Error ##################
######################################################################

tracking_error = x_real_mpc[:T_sim,:] - x_opt[:T_sim,:]
# Calculate the absolute tracking error
abs_tracking_error = np.abs(x_real_mpc[:T_sim,:]- x_opt[:T_sim,:])

bound1 = 0.05 * np.ones(len(abs_tracking_error))
bound2 = -bound1


# Plot the tracking error for each state
fig, axs = plt.subplots(4, 2, figsize=(12, 12), sharex=True, sharey=True,gridspec_kw={'height_ratios': [1, 1, 1, 1]})
plt.subplots_adjust(hspace=0.4, wspace=0.05)
fig.suptitle('Tracking Error for Each State "MPC Without Delta_u"', fontsize=16)

num_states = 8  # Assuming there are 8 states

for i in range(4):
    for j in range(2):
        state_index = i * 2 + j
        if state_index < num_states:  # Check if the state index is within bounds
            axs[i, j].plot(tracking_error[:, state_index], '--b', label=f'Tracking Error State {state_index + 1}')
            axs[i, j].plot(bound1, '--r', label='Error bounds')
            axs[i, j].plot(bound2, '--r')
            axs[i, j].grid()
            axs[i, j].legend()
            
            # Set specific limits for the smaller subplot
            axs[i, j].set_ylim([-0.4, 0.4])

# Set labels for the last row and last column
for j in range(2):
    axs[3, j].set_xlabel('Time')

for i in range(4):
    axs[i, 0].set_ylabel('Tracking Error')

plt.show()


# Plot the tracking error in logarithmic scale
fig, axs = plt.subplots(4, 2, figsize=(10, 10), sharex=True)
fig.suptitle('Tracking Error for Each State "MPC Without Delta_u"', fontsize=16, y = 0.95)
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
######## Part VIII : Animation of Perturbed Optimal Trajectory #######
######################################################################
sim.animate_quadrotor(x_real_mpc, u_real_mpc, dt, params)