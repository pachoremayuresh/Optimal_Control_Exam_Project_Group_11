import numpy as np

class Quadrotor:
    def __init__(self,):
        # Simulation Parameters
        self.tf = 10 # Final time (end of simulation), set to 10.
        self.dt = 0.001  # Time step size 1e-3       
        self.T  = self.tf / self.dt  # Total number of time steps

        # Model Parameters
        self.M = 0.028
        self.m = 0.04
        self.J = 0.001
        self.g = 9.81
        self.L = 0.2
        self.k = 0.05

        pass

    def Quadrotor_Dyanmics(self, state, control_input):
        """
        Computes the dynamics of the quadrotor.
        
        Parameters:
        state: np.array - The current state of the quadrotor [x_pos, y_pos, alpha, theta, vel_x, vel_y, ang_vel_alpha, ang_vel_theta]
        control_input: np.array - The control inputs [Fs, Fd] (Total or symmetric force, Differential force due to the rotors)
        
        Returns:
        np.array - The derivatives of the state
        """
        x_pos, y_pos, alpha, theta, vel_x, vel_y, ang_vel_alpha, ang_vel_theta = state
        Fs, Fd = control_input
        
        #Euler Integration
        next_x_pos = x_pos + vel_x * self.dt
        next_y_pos = y_pos + vel_y * self.dt
        next_alpha = alpha + ang_vel_alpha * self.dt
        next_theta = theta + ang_vel_theta * self.dt



        # Dynamics equations
        dvel_x = (self.m * self.L * ang_vel_alpha**2 * np.sin(alpha) - Fs * (np.sin(theta) - self.m/self.M * np.sin(alpha - theta) * np.cos(alpha))) / (self.M + self.m)
        dvel_y = (-self.m * self.L * ang_vel_alpha**2 * np.cos(alpha) + Fs * (np.cos(theta) + self.m/self.M * np.sin(alpha - theta) * np.sin(alpha)) - (self.M + self.m)* self.g) / (self.M + self.m)
        dang_vel_alpha = (-Fs * np.sin(alpha-theta)) / (self.M * self.L)
        dang_vel_theta = (Fd * self.k) / self.J

        # Update velocities
        # Note: The control inputs Fs and Fd are assumed to be constant over a timestep
        next_vel_x = vel_x + dvel_x * self.dt 
        next_vel_y = vel_y + dvel_y * self.dt
        next_ang_vel_alpha = ang_vel_alpha + dang_vel_alpha * self.dt
        next_ang_vel_theta = ang_vel_theta + dang_vel_theta * self.dt

        return np.array([next_x_pos, next_y_pos, next_alpha, next_theta, next_vel_x, next_vel_y, next_ang_vel_alpha, next_ang_vel_theta])


    def jacobian(self, state, control_input):
        """
        Computes the Jacobian matrices A and B for the quadrotor dynamics.

        Parameters:
        state: np.array - The current state of the quadrotor [x_pos, y_pos, alpha, theta, vel_x, vel_y, ang_vel_alpha, ang_vel_theta]
        control_input: np.array - The control inputs [Fs, Fd] (Total or symmetric force, Differential force due to the rotors)
        
        Returns:
        A: np.array - The Jacobian matrix with respect to the state
        B: np.array - The Jacobian matrix with respect to the control input

        """
        x_pos, y_pos, alpha, theta, vel_x, vel_y, ang_vel_alpha, ang_vel_theta = state
        Fs, Fd = control_input

        par_vel_x_alpha     = self.dt * (self.m * self.L * ang_vel_alpha**2 * np.cos(alpha) + Fs * self.m/self.M * np.cos(2*alpha-theta)) / (self.M + self.m)
        par_vel_x_theta     = - Fs * self.dt* (np.cos(theta) + self.m/self.M * np.cos(alpha - theta) * np.cos(alpha)) / (self.M + self.m)
        par_vel_x_AValpha   = 2 * self.dt* self.m * self.L * ang_vel_alpha * np.sin(alpha)/(self.m + self.M)
             
        par_vel_y_alpha     = self.dt * (self.m * self.L * ang_vel_alpha**2 * np.sin(alpha) + Fs * self.m/self.M * np.sin(2*alpha-theta)) / (self.M + self.m)
        par_vel_y_theta     = - Fs * self.dt* (np.sin(theta) + self.m/self.M * np.cos(alpha - theta) * np.sin(alpha)) / (self.M + self.m)
        par_vel_y_AValpha   = - self.dt * (2 * self.m * self.L * ang_vel_alpha * np.cos(alpha) ) / (self.M + self.m)

        par_AV_a_alpha = - self.dt * Fs * np.cos(alpha - theta) / (self.M * self.L)
        par_AV_a_theta = + self.dt * Fs * np.cos(alpha - theta) / (self.M * self.L)
           
        
        # Compute Jacobian Matrix A
        A = np.array([
            [1,0,               0,               0,  self.dt,       0,                  0,        0],
            [0,1,               0,               0,        0, self.dt,                  0,        0],
            [0,0,               1,               0,        0,       0,            self.dt,        0],
            [0,0,               0,               1,        0,       0,                  0,  self.dt],
            [0,0, par_vel_x_alpha, par_vel_x_theta,        1,       0,  par_vel_x_AValpha,        0],
            [0,0, par_vel_y_alpha, par_vel_y_theta,        0,       1,  par_vel_y_AValpha,        0],
            [0,0,  par_AV_a_alpha,  par_AV_a_theta,        0,       0,                  1,        0],
            [0,0,               0,               0,        0,       0,                  0,        1]
        ])

        par_vel_x_Fs = - self.dt * ( np.sin(theta) - self.m / self.M *np.sin(alpha-theta) * np.cos(alpha) )/ (self.M + self.m)
        par_vel_y_Fs = self.dt*( np.cos (theta) + self.m / self.M * np.sin(alpha- theta) * np.sin(alpha) )/ (self.M + self.m)
        par_AV_a_Fs = - self.dt * np.sin(alpha - theta) / (self.M * self.L)
        par_AV_t_Fd = self.dt * self.k / self.J

        # Compute Jacobian Matrix B
        B = np.array([
            [           0,           0],
            [           0,           0],
            [           0,           0],
            [           0,           0],
            [par_vel_x_Fs,           0],
            [par_vel_y_Fs,           0],
            [ par_AV_a_Fs,           0],
            [           0, par_AV_t_Fd],
        ])

        return A, B

    