
import cvxpy as cp
from Quadrotor import Quadrotor
import scipy.linalg

Drone = Quadrotor()

def solver_linear_mpc(x_d, u_d, A, B, Q, R, Qf, x0, u_min_constraints, u_max_constraints, x_min_constraints, x_max_constraints, T_pred = 5):
    """
        Linear MPC solver - Constrained LQR

        Parameters:
        x_d: np.array - Desired state trajectory
        u_d: np.array - Desired control input trajectory
        A: np.array - State transition matrix
        B: np.array - Control input matrix
        Q: np.array - State cost matrix
        R: np.array - Control cost matrix
        Qf: np.array - Final state cost matrix
        x0: np.array - Initial state
        U_constraints: tuple - Control input constraints (umin, umax)
        X_constraints: tuple - State constraints (xmin, xmax)
        T_pred: int - Prediction horizon

        Returns:
        u_opt: np.array - Optimal control input sequence
        x_pred: np.array - Predicted state trajectory
        u_pred: np.array - Predicted control input trajectory

    """
    P = scipy.linalg.solve_discrete_are(A[0,:,:], B[0,:,:], Q, R)
    x0 = x0.squeeze()

    ns = Drone.ns  # Number of states
    nu = Drone.nu  # Number of control inputs

    x_mpc = cp.Variable((T_pred, ns))
    u_mpc = cp.Variable((T_pred, nu))

    cost = 0
    constraints = []

    for t in range(T_pred - 1):
        cost += cp.quad_form((x_mpc[t, :] - x_d[t, :]), Q) + cp.quad_form((u_mpc[t, :] - u_d[t, :]), R)
        constraints += [x_mpc[t+1, :] == A[t, :, :]@x_mpc[t, :] + B[t, :, :]@u_mpc[t, :], #dynamics constraint
                        
                        # control input constraints
                        u_mpc[t, 0] <= u_max_constraints[0],  
                        u_mpc[t, 0] >= u_min_constraints[0],  
                        u_mpc[t, 1] <= u_max_constraints[1],
                        u_mpc[t, 1] >= u_min_constraints[1],

                        # state constraints
                        x_mpc[t, 0] <= x_max_constraints[0],
                        x_mpc[t, 0] >= x_min_constraints[0],
                        x_mpc[t, 1] <= x_max_constraints[1],
                        x_mpc[t, 1] >= x_min_constraints[1],
                        x_mpc[t, 2] <= x_max_constraints[2],
                        x_mpc[t, 2] >= x_min_constraints[2],
                        x_mpc[t, 3] <= x_max_constraints[3],
                        x_mpc[t, 3] >= x_min_constraints[3],
                        x_mpc[t, 4] <= x_max_constraints[4],
                        x_mpc[t, 4] >= x_min_constraints[4],
                        x_mpc[t, 5] <= x_max_constraints[5],
                        x_mpc[t, 5] >= x_min_constraints[5],
                        x_mpc[t, 6] <= x_max_constraints[6],
                        x_mpc[t, 6] >= x_min_constraints[6],
                        x_mpc[t, 7] <= x_max_constraints[7],
                        x_mpc[t, 7] >= x_min_constraints[7]
        ]

    cost += cp.quad_form((x_mpc[T_pred - 1, :] - x_d[T_pred - 1, :]), P)
    constraints += [x_mpc[0, :] == x0]  # initial condition constraint
                        
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    if prob.status == "infeasible":
    # Otherwise, problem.value is inf or -inf, respectively.
        print("Infeasible problem! CHECK YOUR CONSTRAINTS!!!")

    return u_mpc[0, :].value, x_mpc.value, u_mpc.value