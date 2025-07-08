import numpy as np
import matplotlib.pyplot as plt

from Quadrotor import Quadrotor
import Ref_Trajectory as rf
import Affine_LQR as LQR
import Cost_Function as cost
import Simulator as sim

Drone = Quadrotor()

def gen_trajectory(trajectory_type, loop_type):
    ######################################################################
    ############### Part I : Initialization of parameters ################
    ######################################################################

    term_cond = 1e-6
    max_iters = 100

    J = np.zeros(max_iters) # Cost function values

    # Affine LQR matrices
    Qkt = np.zeros((Drone.T, Drone.ns, Drone.ns))
    Rkt = np.zeros((Drone.T, Drone.nu, Drone.nu))   
    Skt = np.zeros((Drone.T, Drone.nu, Drone.ns))
    qkt = np.zeros((Drone.T, Drone.ns))
    rkt = np.zeros((Drone.T, Drone.nu))

    # State input trajectory
    x = np.zeros((Drone.T, Drone.ns))
    u = np.zeros((Drone.T, Drone.nu))
    x_inter = np.zeros((max_iters, Drone.T, Drone.ns))
    u_inter = np.zeros((max_iters, Drone.T, Drone.nu))

    #reference trajectory
    if trajectory_type == "Step":
        x_ref, u_ref = rf.step_ref_trajectory(Drone.T)
    elif trajectory_type == "Smooth":
        x_ref, u_ref = rf.smooth_ref_trajectory(Drone.T)
    else:
        print("Please select a reference trajectory type: Step or Smooth")

    # Jacobian matrices
    At = np.zeros((Drone.T, Drone.ns, Drone.ns))
    Bt = np.zeros((Drone.T, Drone.ns, Drone.nu))

    # Descent direction matrices
    du = np.zeros((Drone.nu, Drone.T))
    dx = np.zeros((Drone.ns, Drone.T))
    dx0 = np.zeros((Drone.ns))

    Pt = np.zeros((Drone.T, Drone.ns, Drone.ns))
    pt = np.zeros((Drone.T, Drone.ns))
    Kt = np.zeros((Drone.T, Drone.nu, Drone.ns))
    sigmat = np.zeros((Drone.T, Drone.nu))
    ct = np.zeros((Drone.T, Drone.ns))

    descent = np.zeros(max_iters)
    descent_arm = np.zeros(max_iters)

    '''function for running the nonlinear dynamic and return the states'''
    def run_nonlinear_dynamics(initial_state, control_inputs, dt, params):
        num_steps = len(control_inputs)
        states = np.zeros((num_steps, len(initial_state)))
        states[0, :] = initial_state
        for i in range(1, num_steps):
            states[i, :] = Drone.Quadrotor_Dyanmics(states[i-1, :], control_inputs[i-1,:])
        return states

######################################################################
############### Part II : Newton's Method Algorithm ##################
######################################################################

## Step 0: Initial Input Guess
    u = np.column_stack((np.linspace((Drone.M + Drone.m) * Drone.g, (Drone.M + Drone.m) * Drone.g, Drone.T),np.linspace(0, 0, Drone.T))) # Control input guess
    x = run_nonlinear_dynamics(rf.state_eq1, u, Drone.dt, [Drone.M, Drone.m, Drone.J, Drone.g, Drone.L, Drone.k])  # Initial state guess

    for k in range(max_iters):

    ## Step 1: Descent Direction Calculation
        # Extract the neccessary matrices and the cost function to find the descent direction
        for t in range(Drone.T-1):
            dcost_dstate, dcost_dcontrol_input, l11_dxx, l22_duu, l12_dxu, Qt, Rt, St, lcost = cost.stage_cost(x[t,:], u[t,:], x_ref[t,:], u_ref[t,:], t)    
            Qkt[t,:,:] = Qt
            Rkt[t,:,:] = Rt
            Skt[t,:,:] = St
            qkt[t,:] = dcost_dstate
            rkt[t,:] = dcost_dcontrol_input
            J[k] += lcost #cost

            At[t,:,:], Bt[t,:,:] = Drone.Jacobian(x[t,:], u[t,:])  # Jacobian matrices

        lcost_T, dcost_T,_,Pf = cost.terminal_cost(x[-1,:], x_ref[-1,:])  # Terminal cost
        qkt[-1,:] = np.zeros([1, Drone.ns]) # No gradient at the terminal state
        J[k] += lcost_T

        # The Explicit Calculation of the descent direction by affine LQR
        Kt, sigmat, Pt, dx, du = LQR.ltv_LQR(At, Bt, Qkt, Rkt, Skt, Drone.T, dx0, Pf, qkt, rkt, qkt[-1,:])

        _lambda = np.zeros((Drone.T, Drone.ns))
        dJ = np.zeros((Drone.T, Drone.nu))
        _lambda_temp = cost.terminal_cost(x[-1,:], x_ref[-1,:])[1]
        _lambda[-1,:] = _lambda_temp.squeeze()



    ## Step 2: Computation of dJ(u)

        for t in reversed(range(Drone.T-1)):
            at, bt = cost.stage_cost(x[t,:], u[t,:], x_ref[t,:], u_ref[t,:], t)[:2]

            _lambda_temp = At[t,:,:].T @ _lambda[t+1,:].T + at    # Cost equation
            dJ_temp = Bt[t,:,:].T @ _lambda[t+1,:].T + bt           # Gradient of the cost function w.r.t. control input

            _lambda[t,:] = _lambda_temp.squeeze()
            dJ[t,:] = dJ_temp.squeeze()

            descent_arm[k] += dJ[t,:] @ du[:,t]
            descent[k] += du[:,t] @ du[:,t]
            
        step_sizes = []
        costs_armijo = []

        beta_armijo = 0.8
        c_armijo = 0.5
        step_size = 1



    ## Step 3: Armijo rule
        max_its = 100
        iter = 0

        while iter < max_its:
            # for i_armijo in range(100):
            #temp solution update
            x_temp = np.zeros((Drone.T, Drone.ns))
            u_temp = np.zeros((Drone.T, Drone.nu))
            x_temp[0,:] = rf.state_eq1

            #update the control input and state

            for t in range(Drone.T-1):
                if loop_type == "open":
                    u_temp[t,:] = u[t,:] + step_size * du[:,t]
                elif loop_type == "closed":
                    u_temp[t,:] = u[t,:] + Kt[t,:,:] @ (x_temp[t,:] - x[t,:]).T + step_size * sigmat[t,:]
                else:
                    print("Please select a loop type: open or closed")
                x_temp[t+1,:] = Drone.Quadrotor_Dyanmics(x_temp[t,:], u_temp[t,:])
            
            J_temp = 0
            for t in range(Drone.T-1):
                temp_cost = 0
                temp_cost = cost.stage_cost(x_temp[t,:], u_temp[t,:], x_ref[t,:], u_ref[t,:], t)[8]
                J_temp += temp_cost
            temp_cost = cost.terminal_cost(x_temp[-1,:], x_ref[-1,:])[0]
            J_temp += temp_cost
            step_sizes.append(step_size) #saving step size
            costs_armijo.append(J_temp) #saving cost function value
            if J_temp > J[k] + c_armijo * step_size * descent_arm[k]:
                # update the step size
                step_size *= beta_armijo
            else:
                print("Armijo Stepsize = {}".format(step_size))
                break
        
    
    
    ## Step 4: Armijo Plot
        if k%1 == 0:
            steps = np.linspace(0,1,int(20))
            costs = np.zeros(len(steps))

            for i in range(len(steps)):
                step = steps[i]

                # temp solution update
                x_temp = np.zeros((Drone.T, Drone.ns))
                u_temp = np.zeros((Drone.T, Drone.nu))
                x_temp[0,:] = rf.state_eq1

                for t in range(Drone.T-1):
                    if loop_type == "open":
                        u_temp[t,:] = u[t,:] + step * du[:,t]
                    elif loop_type == "closed":
                        u_temp[t,:] = u[t,:] + Kt[t,:,:] @ (x_temp[t,:] - x[t,:]) + step * sigmat[t,:]
                    else:
                        print("Please select a loop type: open or closed")
                    x_temp[t+1,:] = Drone.Quadrotor_Dyanmics(x_temp[t,:], u_temp[t,:])

                J_temp = 0
                for t in range(Drone.T-1):
                    temp_cost = cost.stage_cost(x_temp[t,:], u_temp[t,:], x_ref[t,:], u_ref[t,:], t)[8]
                    J_temp += temp_cost
                temp_cost = cost.terminal_cost(x_temp[-1,:], x_ref[-1,:])[0]
                J_temp += temp_cost
                costs[i] = J_temp
                
            plt.figure(1)
            plt.clf()
            plt.plot(steps, costs, color='g', label='$J(\\mathbf{u}^k + stepsize*d^k)$')
            plt.plot(steps, J[k] + descent_arm[k]*steps, color='r', label='$J(\\mathbf{u}^k) + stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
            plt.plot(steps, J[k] + c_armijo*descent_arm[k]*steps, color='g', linestyle='dashed', label='$J(\\mathbf{u}^k) + stepsize*c*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
            plt.scatter(step_sizes, costs_armijo, marker='*') # plot the tested stepsize
            plt.grid()
            plt.xlabel('stepsize')
            plt.legend()
            plt.draw()
            plt.show()



    ## Step 5: Update the control input and state

        x_temp = np.zeros((Drone.T, Drone.ns))
        u_temp = np.zeros((Drone.T, Drone.nu))

        x_temp[0,:] = rf.state_eq1

        for t in range(Drone.T-1):
            if loop_type == "open":
                u_temp[t,:] = u[t,:] + step_size * du[:,t]
            elif loop_type == "closed":
                u_temp[t,:] = u[t,:] + Kt[t,:,:] @ (x_temp[t,:] - x[t,:]) + step_size * sigmat[t,:]
            else:
                print("Please select a loop type: open or closed")
            x_temp[t+1,:] = Drone.Quadrotor_Dyanmics(x_temp[t,:], u_temp[t,:])
        x[:,:] = x_temp
        u[:,:] = u_temp
        x_inter[k,:,:] = x
        u_inter[k,:,:] = u


    ## Step 6: Termination Condition
        print('Iter = {}\t Descent = {:.3e}\t Cost = {:.3e}'.format(k,descent[k], J[k]))

        if descent[k] <= term_cond:
            max_iters = k
            print('MaxIter = {}'.format(k))
            break

    return u,x,u_ref, x_ref, max_iters,descent, J, Drone.dt,[Drone.M, Drone.m, Drone.J, Drone.g, Drone.L, Drone.k], x_inter, u_inter, Drone.T



if __name__ == "__main__":

    trajectory_type = "Step"  # Choose between "Step" or "Smooth"
    loop_type = "open"  # Choose between "open" or "closed"

    u, x, u_ref, x_ref, max_iters, descent, J, dt, params, x_inter, u_inter, T = gen_trajectory(trajectory_type, loop_type)

    ## Step 7: Plotting the results
    # 7.1  Optimal trajectory and desired curve
    # Plot each state in a separate subplot and compare it with the refrence curve
    fig, axs = plt.subplots(4, 2, figsize=(10, 20), sharex=True,dpi=100)
    fig.suptitle('Optimal Trajectory Vs Reference Curve State', fontsize=16, y = 0.95)
    i = 0
    j = 0
    k = 1
    l = 0
    while i < 4:
        while j < 2:
            axs[i,j].plot(x[:, l],'b', label = 'State')
            axs[i,j].plot(x_ref[:, l],'--r', label = 'State_ref')
            axs[i,j].legend(loc = "upper left")
            axs[i,j].set_title(f'State {i + j + k}', fontsize=12)
            axs[i,j].grid()
            j = j + 1
            l = l + 1
            pass
        i = i + 1
        k = k + 1
        j = 0
        pass

    plt.show()
    
    for i in range(8):
        plt.figure()
        plt.plot(x[:, i],'b', label=f'State {i + 1}')
        plt.plot(x_ref[:, i],'--r',label=f'State_ref {i + 1}')
        plt.xlabel('Time')
        plt.grid()
        plt.legend()
    for i in range(2):
        plt.figure()    
        plt.plot(u[:-1,i],'b', label=f'input {i + 1}')
        plt.plot(u_ref[:,i],'--r', label=f'input {i + 1}')
        plt.xlabel('Time')
        plt.grid()
        plt.legend()
    plt.show()


        
    ###7.2 Optimal trajectory, desired curve and few intermediate trajectories for stat1 and atate2.
 
    fig, axs = plt.subplots(2, 1, figsize=(10, 20), sharex=True)
    fig.suptitle('Optimal & IntermediateTrajectories Vs RefrenceCurve for Stat1&State2', fontsize=16)
    for i in range(2):
        axs[i].plot(x[:, i],'b', label=f'opt_State {i + 1}')
        axs[i].plot(x_ref[:, i],'--r', label=f'State_ref {i + 1}')
        axs[i].set_ylabel(f'State {i + 1}')
        axs[i].grid()
    axs[1].set_xlabel('Time')
    for ax in axs:
        ax.legend()
    for k in range (round(3)):
        # Plot each state in a separate subplot and compare it with the refrence curve
        for i in range(2):
            axs[i].plot(x_inter[k,:, i], label=f'iter {k}') 
            plt.grid()
    plt.grid()
    for ax in axs[0:1]:
      ax.legend()
    plt.grid()
    plt.show()

    # 7.3  Norm of the descent direction along iterations (semi-logarithmic scale)>
    plt.figure('descent direction')
    plt.title('Descent Direction', fontsize=16)
    plt.plot(np.arange(max_iters), descent[:max_iters])
    plt.xlabel('$k$')
    plt.ylabel('||$\\nabla J(\\mathbf{u}^k)||$')
    plt.yscale('log')
    plt.grid()
    plt.show()

    # 7.4  Cost along iterations (semi-logarithmic scale).
    plt.figure('cost')
    plt.title('Cost', fontsize=16)
    plt.plot(np.arange(max_iters), J[:max_iters])
    plt.xlabel('$k$')
    plt.ylabel('$J(\\mathbf{u}^k)$')
    plt.yscale('log')
    plt.grid()
    plt.show()

    # 7.5 Animation
    print("the optimal control sequence is:")
    print(u)
    print("**************")
    simulated_states = x
    sim.animate_quadrotor(simulated_states, u, dt, params)