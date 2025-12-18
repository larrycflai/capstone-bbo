# Model Card: Bayesian Optimisation for BBO Capstone

## 1. Overview
*   **Model Name:** Gaussian Process Regressor (GPR) with Adaptive Kernels
*   **Version:** 1.0 (Round 10 Snapshot)
*   **Model Type:** Bayesian Optimisation (Maximisation)
*   **Library:** `scikit-learn` (GaussianProcessRegressor)
*   **Author:** Larry Lai
*   **Date:** 18.12.2025
*   **License:** MIT / Academic Use Only

## 2. Intended Use
*   **Primary Task:** Optimisation of **eight synthetic black-box functions** simulating real-world tasks (e.g., radiation detection, robot control, drug discovery).
*   **Input Domain:** Continuous values strictly bounded within the **Unit Hypercube** $^d$, where $d \in \{2, 3, 4, 5, 6, 8\}$ corresponds to the specific function's dimensionality.
*   **Output:** Maximisation of the target scalar value $Y$.
*   **Out-of-Scope Use Cases:**
    *   **Real-time Control:** Not suitable for high-frequency trading or sub-second robotics due to the cubic computational cost ($O(N^3$)) of re-training the Gaussian Process at every step.
    *   **Ground Truth Simulation:** This model should not be treated as a perfect surrogate for the underlying functions. Due to data sparsity (~19 points per function), the model's uncertainty ($\sigma$) remains high in unobserved regions.

## 3. Model Details & Strategy Evolution
The optimisation strategy evolved iteratively over 10 weeks to balance **Exploration** (gathering data in unknown regions) and **Exploitation** (refining the best known results).

### Core Architecture
*   **Surrogate Model:** Gaussian Process Regressor (GPR).
*   **Acquisition Function:** Expected Improvement (EI) was the primary driver, with Adaptive UCB used in early rounds [6].
*   **Optimisation Method:** L-BFGS-B with random restarts (10–50 restarts) to escape local optima in the acquisition surface.

### Strategy Evolution Log
| Phase | Weeks | Strategy Description | Rationale |
| :--- | :--- | :--- | :--- |
| **Initialization** | Wk 1 | **Random Search** | Established an initial scatter of points to understand the landscape and avoid initial bias. |
| **Early Exploration** | Wk 2 | **Adaptive Mix (EI + UCB)** | Switched poorly performing functions to **UCB** (Exploration) to force the model into new regions, while exploiting promising functions with **EI** [8]. |
| **Smoothness Assumption** | Wk 3–4 | **RBF Kernel** | Switched to Radial Basis Function (RBF) to test if the objective functions were infinitely smooth/simple. |
| **Noise Modelling** | Wk 5 | **RBF + `WhiteKernel`** | **Critical Pivot:** Introduced noise modeling (`noise_level=0.1`) to handle observational noise. The functions exhibited stochastic behavior (variance in $Y$), requiring the model to distinguish between signal and noise. |
| **Robust Refinement** | Wk 6–10 | **Matern ($\nu=2.5$) + `WhiteKernel`** | Reverted to the Matern kernel for flexibility (handling non-smooth surfaces) while maintaining the `WhiteKernel`. Increased exploration parameter ($\xi = 0.05$) in final rounds to escape local optima. |

## 4. Performance & Metrics
Performance is measured by the **cumulative maximum $Y$** found for each function.

*   **Metric:** Absolute Maximisation ($Y_{max}$).
*   **Data Count:** ~19 observed points per function (Initial 10 + 9 weekly queries).
*   **Key Results (Snapshots from Week 8/9 Logs):**
    *   **High Performance:** Function 5 (4D) achieved high-magnitude results, jumping from $Y \approx 1,645$ (Round 7) to $Y \approx 8,662$ (Round 8) after the strategy correctly exploited the upper boundary of the hypercube ([1.0, 1.0, 1.0, 1.0]).
    *   **High-Dimensional Challenge:** Function 8 (8D) proved difficult due to the "Curse of Dimensionality." With only ~19 points covering an 8-dimensional volume, the search space remains largely unexplored.

## 5. Assumptions and Limitations
*   **Assumption of Noise:** The model assumes the target functions contain **aleatoric (observational) noise**. This hypothesis was confirmed when the model stability improved after introducing the `WhiteKernel` in Week 5.
*   **Boundary Constraints:** The model strictly enforces bounds of $[0.0, 0.999999]$. Values outside this range are invalid and truncated.
*   **Sparsity Limitation:** With only ~19 data points, the model's posterior variance in high-dimensional spaces (6D, 8D) remains significant. The global maximum for F6, F7, and F8 likely resides in unobserved regions.

## 6. Ethical Considerations and Transparency
*   **Reproducibility:** A **Datasheet** documenting the data collection process is available [here](./DATASHEET.md).
*   **Transparency:** All query decisions were automated using the code provided in this repository, removing human bias from the selection process (except for the high-level choice of Kernels).
*   **Bias:** The Random Search initialization (Week 1) determines the trajectory of the optimisation. Different random seeds could lead to significantly different local optima convergence.
