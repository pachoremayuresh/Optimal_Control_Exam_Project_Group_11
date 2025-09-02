# Optimal Control of a Quadrotor with Suspended Load

<p align="justify">
This project investigates optimal control strategies for a quadrotor equipped with a suspended load, a system characterized by nonlinear and highly coupled dynamics. The study focuses on three methods: Newton’s method for trajectory generation, Linear Quadratic Regulator (LQR) for trajectory tracking, and Model Predictive Control (MPC) for predictive constrained control.

Newton’s method is applied to compute optimal state-input trajectories between equilibrium points, exploiting second-order information for fast convergence. Both step-based and smooth reference trajectories are evaluated, with results showing that smooth references improve convergence speed and accuracy. Based on these trajectories, the system is linearized and an LQR controller is designed to provide robust tracking and disturbance rejection. Finally, an MPC framework is developed to handle system and input constraints while ensuring smooth control actions. Simulations demonstrate that MPC achieves superior tracking in the presence of disturbances, although with higher computational cost.

The comparative analysis highlights the complementary nature of these approaches: Newton’s method is effective for trajectory optimization, LQR ensures reliable real-time performance, and MPC provides flexibility under constraints. These findings contribute to the development of advanced control solutions relevant to applications in aerial robotics and autonomous systems.

</p>

---

## Task 0 – Problem Setup
Begin with setting up the problem:
- In `Quadrotor.py`, you'll find the dynamics function, Jacobians **A** & **B**, and the parameters.

---

## Task 1 – Trajectory Generation (I)
- To define a reference curve between two equilibrium points, open `Ref_Trajectory.py`.
- Initially, the equilibrium points are `(xp1, yp1) = (0, 0)` and `(xp2, yp2) = (1, 1)`. Feel free to choose other values.
- After setting the equilibrium points, proceed to `Newtons_Method.py` and run the code. 
- The default reference curve is a step function. The algorithm will find the optimal trajectory after a few iterations.
- **Outputs include:**
  - Optimal trajectory
  - Desired curve
  - Intermediate trajectories
  - Armijo descent direction plot
  - Cost and norm plots along iterations

---

## Task 2 – Trajectory Generation (II)
- Generate a smooth state-input curve and perform trajectory generation (Task 1) on this new desired trajectory.
- After setting the equilibrium points, proceed to `Newtons_Method.py` and run the code. 
- In the main loop, `trajectory_type = "Step"`. Change it to `"Smooth"`.  
- The algorithm will find the optimal trajectory after a few iterations.
- Run the code to observe the results.

---

## Task 3 – Trajectory Tracking via LQR
- This task uses the **LQR algorithm** for optimal feedback control to track the reference trajectory.
- Open `Traj_Tracking.py` and set the `perturbed_initial_state` under **Set the Initial Perturbed State**.
- The code outputs comparisons and animations showing trajectories **with and without LQR**.

---

## Task 4 – Trajectory Tracking via MPC
- This task involves using an **MPC algorithm** for trajectory tracking.
- Open `main_MPC.py` and under **Set the Initial Perturbed State**, you can set or create your own `initial_state_perturbed`.
- To see how MPC behaves if perturbed **during tracking**, uncomment lines under:
  - *Add external disturbance III*
  - *Add external disturbance IV*
- Run the code to check:
  - System trajectory plots
  - Tracking error plots
  - Animations



Feel free to experiment and modify the parameters to explore different aspects of the control algorithms. Enjoy your journey into quadrotor control!

## 🤝 Contributing

Contributions are welcome!  
If you’d like to improve this repository, here are some ways to help:

- 🐛 **Report Issues**: Found a bug or something unclear? [Open an issue](../../issues).
- 💡 **Suggest Features**: Have an idea for improvement? Share it in an issue.
- 🔧 **Submit Pull Requests**: Want to fix a bug, improve documentation, or add new functionality? Fork the repo and open a PR.
- 📝 **Improve Documentation**: Even small corrections (typos, better explanations) are valuable.

Please make sure to:
1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/my-update`)  
3. Commit your changes (`git commit -m "Description of changes"`)  
4. Push to your branch (`git push origin feature/my-update`)  
5. Open a Pull Request  

---

## ⭐ Support

If you find this project useful, consider giving it a **star** on GitHub — it helps others discover the repo!

---

## 📬 Contact

For questions or suggestions, feel free to [open an issue](../../issues) or reach out via discussions if enabled.

## Acknowledgment & Credits

This project was **originally developed by Group 11** as part of the *Optimal Control Exam 2023/24*.  
I, **Mayuresh Pachore**, have **redeveloped and restructured** the code based on the original work to publish it in a more constructive and accessible form.  

A special thanks to the original contributors — **Ammar Garooge** and **Santoshkumar Hankare** — for their valuable work, which laid the foundation for this enhanced version.



You may also cite relevant academic papers here if applicable.

Or, if available, use this BibTeX entry:

```bibtex
@misc{pachoremayuresh2023optimal_control_exam_project_group_11,
  author       = {Pachore, Mayuresh},
  title        = {Optimal Control Exam Project Group 11},
  howpublished = {\url{https://github.com/pachoremayuresh/Optimal\_Control\_Exam\_Project\_Group\_11}},
  year         = {2023},
  note         = {Online; accessed 2 September 2025}
}

