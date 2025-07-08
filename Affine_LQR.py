import numpy as np
from Quadrotor import Quadrotor

Drone = Quadrotor()

nu = Drone.nu  # Number of control inputs
ns = Drone.ns  # Number of states

def ltv_LQR(A, B, Q, R, S, T, x0, Qf, q=None, r=None, qf=None):

    """
    Computes the LQR gain for a linear time-varying system.
    
    Parameters:
    A: np.array - State transition matrix
    B: np.array - Control input matrix
    Q: np.array - State cost matrix
    R: np.array - Control cost matrix
    S: np.array - Cross-term cost matrix
    T: int      - Number of time steps
    x0: np.array - Initial state
    Qf: np.array - Terminal cost matrix
    q: np.array - State cost vector (optional)
    r: np.array - Control cost vector (optional)    
    qf: np.array - Terminal state cost vector (optional)
    
    Returns:
    K: np.array - LQR gain matrix
    sigma: np.array - Feedforward term
    P: np.array - Solution to the Riccati equation
    x: np.array - State trajectory
    u: np.array - Control input trajectory
    
    """
    K = np.zeros(T,nu,ns)
    sigma = np.zeros((T, nu))
    P = np.zeros((T, ns, ns))
    p = np.zeros((T, ns))

    x = np.zeros((ns, T)) # Initial state
    u = np.zeros((nu, T)) # Control input
    
    x[:, 0] = x0  # Set initial state

    P[-1,:,:] = Qf  # Terminal cost matrix
    p[-1,:] = qf

    # Solve Riccati equation backwards in time
    for t in reversed(range(T-1)):
        Qt = Q[t,:,:]
        qt = q[t,:][:, None]
        Rt = R[t,:,:]
        rt = r[t,:][:, None]
        At = A[t,:,:]
        Bt = B[t,:,:]
        St = S[t,:,:]
        Pt = P[t+1,:,:]
        pt = p[t+1,:][:, None]

        Mt_inv = np.linalg.inv(Rt + Bt.T @ Pt @ Bt)
        mt = rt + Bt.T @ pt

        Pt = At.T @ Pt @ At - (Bt.T @ Pt @ At + St).T @ Mt_inv @ (Bt.T @ Pt @ At + St) + Qt
        pt = At.T @ pt - (Bt.T @ Pt @ At + St).T @ Mt_inv @ mt + qt

        P[t,:,:] = Pt
        p[t,:] = pt.squeeze()

    # Compute the LQR gain
    for t in range(T-1):
        Qt = Q[t,:,:]
        qt = q[t,:][:, None]
        Rt = R[t,:,:]
        rt = r[t,:][:, None]
        At = A[t,:,:]
        Bt = B[t,:,:]
        St = S[t,:,:]
        Pt = P[t+1,:,:]
        pt = p[t+1,:][:, None]

        # Check positive definiteness
        Mt_inv = np.linalg.inv(Rt + Bt.T @ Pt @ Bt)
        mt = rt + Bt.T @ pt

        #For other Purposes we could add regularization here
        K[t,:,:] = -Mt_inv @ (Bt.T @ Pt @ At + St)
        sigma[t,:] = (-Mt_inv @ mt).squeeze()

    for t in range(T-1):

        #Trajectory
        u[:, t] = K[t,:,:] @ x[:, t] + sigma[t,:]
        x_p = A[t,:,:] @ x[:, t] + B[t,:,:] @ u[:, t]
        x[:, t+1] = x_p

        
    return K, sigma, P, x, u