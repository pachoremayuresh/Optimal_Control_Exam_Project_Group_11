# Optimal Control of a Quadrotor with Suspended Load

This project was a part of Optimal Control Exam 2023/24 it was done by Group 11.  
**Group Members:** Mayuresh Pachore, Ammar Garooge, Santoshkumar Hankare  

This is an enhanced version of the project with well-structured code for the exam.

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



You may also cite relevant academic papers here if applicable.

Or, if available, use this BibTeX entry:

```bibtex
@misc{pachore2024quadrotor,
  author = {Pachore Mayuresh},
  title = {Optimal Control of a Quadrotor with Suspended Load},
  year = {2024},
  howpublished = {GitHub repository},
  note = {https://github.com/pachoremayuresh/Optimal_Control_Exam_Project_Group_11},
}

