# Optimal Control of a Quadrotor with Suspended Load

This project was a part of Optimal Control Exam 2023/24 it was done by Group 11.
Group Members:- Mayuresh Pachore, Ammar Garooge, Santoshkumar Hankare

This is a enhanced version of the project well structured code for the exam.

Task 0 – Problem Setup:
    Begin with setting up the problem:
    - In "Quadrotor.py", you'll find the dynamics function, Jacobians A & B, and the parameters.


Task 1 – Trajectory Generation (I):
    - To define a reference curve between two equilibrium points, open "Ref_Trajectory.py".
    - Initially, the equilibrium points are (xp1, yp1) = (0, 0) and (xp2, yp2) = (1, 1). Feel free to choose other values.
    - After setting the equilibrium points, proceed to "Newtons_Method.py" and run the code. 
    - The default reference curve is a step function. The algorithm will find the optimal trajectory after a few iterations.
    - Outputs include optimal trajectory, desired curve, intermediate trajectories, Armijo descent direction plot, and cost and norm plots along iterations.


Task 2 – Trajectory Generation (II):
    - Generate a smooth state-input curve and perform trajectory generation (Task 1) on this new desired trajectory.
    - After setting the equilibrium points, proceed to "Newtons_Method.py" and run the code. 
    - The default reference curve is a step function.In the main loop trajectory_type = "Step". Change is to "Smooth". The algorithm will find the optimal trajectory after a few iterations. 
    - Run the code to observe the results.

Task 3 – Trajectory Tracking via LQR:
    - This task uses the LQR algorithm for optimal feedback control to track the reference trajectory.
    - Open "Traj_Tracking.py" and set the "perturbed_initial_state" under "Set the Initial Perturbed State."
    - The code outputs comparisons and animations showing trajectories with and without LQR.

Task 4 – Trajectory Tracking via MPC:
    - This task involves using an MPC algorithm for trajectory tracking.
    - Open "main_MPC.py" and under "Set the Initial Perturbed State", you can set or create your own "initial_state_perturbed"
    - If you want to see how MPC behaves if the Perturbed in between while tracking uncomment lines under 'Add external disturbance III' and 'Add external disturbance IV'.             
    - Run the code to check plots for system trajectory, tracking error, and animations.


Feel free to experiment and modify the parameters to explore different aspects of the control algorithms. Enjoy your journey into quadrotor control!