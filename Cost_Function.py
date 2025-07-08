import numpy as np
from Quadrotor import Quadrotor

Drone = Quadrotor()

ns = Drone.ns  # Number of states
ni = Drone.nu  # Number of inputs 
T = Drone.T  # Total time for the trajectory

# Cost Matrices to achieve the desired behaviour
# For the trajectroy from (0,0) to (1,1)
Qt1 = 0.001*np.diag([100, 100,100,1,1,1,1,1])
Qt2 = 0.001*np.diag([100, 100,100,10,1,1,1,1])
Rt = 1*np.eye(ni)

# For the trajectory from(0,0,0) to (0,0,np.pi)
# Qt1 = 0.001*np.diag([100, 100,100,1,1,1,1,1])
# Qt2 = 0.001*np.diag([100, 100,100,10,1,1,1,1])
# Rt = 1*np.eye(ni)

Pf = Qt2

# Stage cost function
def stage_cost(state, control_input, state_ref, control_input_ref, time_step):
    """
    Computes the stage cost for the quadrotor.
    
    Parameters:
    state: np.array - The current state of the quadrotor
    control_input: np.array - The current control input
    state_ref: np.array - The reference state
    control_input_ref: np.array - The reference control input
    time_step: float - The time step size
    
    Returns:
    dcost_dstate: np.array - The gradient of the cost with respect to the state
    dcost_dcontrol_input: np.array - The gradient of the cost with respect to the control input
    l11_dxx: np.array - The second derivative of the cost with respect to the state
    l22_duu: np.array - The second derivative of the cost with respect to the control input
    l12_dxu: np.array - The cross derivative of the cost with respect to the state and control input
    QQt: np.array - The second derivative of the cost with respect to the state
    RRt: np.array - The second derivative of the cost with respect to the control input
    SSt: np.array - The cross derivative of the cost with respect to the state and control input
    cost: float - The stage cost

    
    """
    state = state.T
    control_input = control_input.T
    state_ref = state_ref.T

    if time_step < T/2:
        Qt = Qt1
    else:
        Qt = Qt2

    #l
    cost = 0.5*(state - state_ref).T @ Qt @ (state - state_ref) + 0.5*(control_input - control_input_ref).T @ Rt @ (control_input - control_input_ref)

    # The gradient of the cost function with respect to the state and control input
    dcost_dstate = Qt @ (state - state_ref)  #at #l1_dx
    dcost_dcontrol_input = Rt @ (control_input - control_input_ref) #bt #l2_du

    # The Hessian of the cost function with respect to the state and control input
    l11_dxx = Qt 
    l22_duu = Rt 
    l12_dxu = np.zeros((ni,ns))

    QQt = l11_dxx
    RRt = l22_duu 
    SSt = l12_dxu

    return dcost_dstate.squeeze(), dcost_dcontrol_input, l11_dxx, l22_duu, l12_dxu, QQt, RRt, SSt, cost



# Terminal cost function
def terminal_cost(state, state_ref):
    """
    Computes the terminal cost for the quadrotor.
    
    Parameters:
    state: np.array - The current state of the quadrotor
    state_ref: np.array - The reference state
    
    Returns:
    cost_T: float - The terminal cost
    dcost_dstate: np.array - The gradient of the terminal cost with respect to the state
    lumda_T: np.array - The Lagrange multiplier for the terminal cost
    Pf: np.array - The terminal cost matrix

    """
    state = state[:, None]
    state_ref = state_ref[:, None]

    cost_T = 0.5 * (state - state_ref).T @ Pf @ (state - state_ref)
    dcost_dstate = Pf @ (state - state_ref)  # Gradient of the terminal cost with respect to the state

    lumda_T = Pf @ (state - state_ref)  # Lagrange multiplier for the terminal cost

    return cost_T.squeeze(), dcost_dstate, lumda_T, Pf
