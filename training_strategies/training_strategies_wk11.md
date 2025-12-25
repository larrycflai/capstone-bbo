# Capstone Training Strategy Logs

## Historical Overview (Weeks 0-7)

| **Week (Round)** | **Primary Acquisition Function** | **GP Kernel Used** | **Core Strategy & Rationale** |
| :--- | :--- | :--- | :--- |
| **0 (Initial)** | Pure Random Search | N/A | Establish initial scatter of points for all functions. |
| **1 (R0 Results)** | Expected Improvement (EI) | Matern | Begin exploiting the existing data. Matern is a general-purpose, robust kernel. |
| **2 (R1 Results)** | **Adaptive Mix** (EI / UCB) | Matern | Introduce **Explore/Exploit Balance**. Functions with poor Week 1 results were switched to **UCB** (Exploration) to find new regions. Functions with good results stayed on **EI** (Exploitation). |
| **3 (R2 Results)** | Expected Improvement (EI) | RBF + C | Shift to RBF (Radial Basis Function) to assume a very **smooth** objective function, testing if the problems are simpler than initially thought. |
| **4 (R3 Results)** | Expected Improvement (EI) | RBF + C | Continued refinement under the smooth RBF assumption. |
| **5 (R4 Results)** | Expected Improvement (EI) | RBF + C + WhiteKernel | Introduced the **WhiteKernel** component to explicitly model **observational noise**. This is a critical technique for robust BBO, acknowledging that the evaluation scores (Y values) might be inherently noisy. |
| **6 (R5 Results)** | Expected Improvement (EI) | **Matern** | Switched back to the more flexible Matern kernel for the final high-confidence search, ensuring the model isn't overly constrained by the RBF's smoothness assumption. |
| **7 (Final)** | Expected Improvement (EI) | **Matern** | The strategy script prepared for Round 7 uses the reliable Matern kernel and focuses purely on **Exploitation (EI)** to find the absolute minimum based on all 6 rounds of data. |

---

## Week 8 Strategy Update
*Mitigating high variance (noise) in evaluations and ensuring the acquisition function finds the globally optimal next query point.*

| **Component** | **Setting/Value** | **Rationale** |
| :--- | :--- | :--- |
| **GPR Kernel (Core)** | ConstantKernel * Matern($\nu=2.5$) | **Robustness:** Matern (with $\nu=2.5$) models non-infinitely smooth functions. This is essential for dealing with the complex landscapes of the synthetic black-box functions. |
| **GPR Kernel (Noise)** | + WhiteKernel(noise_level=0.1) | **Noise Modelling:** Explicitly models the **observational noise** (e.g., simulated sensor error). This improves the model's overall fit by attributing some variance to noise rather than forcing the curve to fit every outlier. |
| **Objective** | **Maximisation** | **Goal Confirmation:** The entire system is designed to find the maximum possible $Y$ score. |
| **Acquisition Function** | **Expected Improvement (EI)** | **Balanced Search:** EI provides a mathematical balance between **exploitation** (querying near the current $x_{best}$) and **exploration** (querying in high-uncertainty areas). |
| **EI $\xi$ Parameter** | $\xi = 0.01$ | **Exploration Tuning:** A small positive value ensures the process maintains some exploration, preventing the optimisation from becoming too greedy, especially in high-dimensional functions (F5-F8). |
| **Acquisition Optimiser** | N_RESTARTS = 10 | **Non-Convexity:** The EI surface has multiple peaks. Running 10 independent optimisation runs (random restarts) significantly increases the probability of finding the true global maximum of the EI function. |

---

## Week 9 Strategy Update
*Aggressive push toward finding the global maximum, introducing higher exploration parameters to escape local optima.*

| **Component** | **Setting/Value** | **Rationale** |
| :--- | :--- | :--- |
| **GPR Kernel (Core)** | ConstantKernel * Matern($\nu=2.5$) | **Robustness:** Matern ($\nu=2.5$) handles non-infinitely smooth, complex objectives, preventing over-certainty. |
| **GPR Kernel (Noise)** | + WhiteKernel(noise_level=0.1) | **Noise Modelling:** Explicitly models the stochasticity (variance) inherent in the evaluation scores. |
| **Objective** | **Maximisation** | **Goal Confirmation:** Confirming the objective is to find the maximum possible $Y$ score. |
| **Acquisition Function** | Expected Improvement (EI) | **Balanced Search:** EI remains the best compromise between exploiting current best scores and exploring uncertain regions. |
| **EI $\xi$ Parameter** | $\xi = \mathbf{0.05}$ | **Aggressive Exploration:** Increased from 0.01 to 0.05. This prioritises points with higher variance (exploration) over pure exploitation, essential for finding breakthrough results in high-dimensional space. |
| **Acquisition Optimiser** | N_RESTARTS = 15 | **Non-Convexity:** Increased restarts from 10 to **15** to further mitigate the risk of getting stuck in a local maximum of the EI function. |

---

## Week 10 Strategy Update
*Reverting to tighter exploitation to refine the best-known regions.*

| Component | Setting | Rationale |
| :--- | :--- | :--- |
| **Method** | **Bayesian Optimisation** | Revert from Random Search (used in Wk9 for coverage) back to BO to leverage the refined data structure. |
| **Kernel** | **Matern($\nu=2.5$) + WhiteKernel** | **Matern** handles the complexity of the functions. **WhiteKernel** is essential because the function outputs contain observational noise; it prevents the model from overfitting when it sees large jumps in data (e.g., F5 $Y \approx 1645 \to 8662$). |
| **Acquisition** | **Expected Improvement (EI)** | Focus on improving over the current $f_{best}$. |
| **Exploration ($\xi$)** | **0.01 (Standard)** | Tighter than the Wk9 strategy. We now want to guide the model toward the best-known regions, only exploring if uncertainty is very high. |
| **Restarts** | **20** | High restarts to ensure the optimizer doesn't get stuck in local optima within the acquisition surface. |

---

## Week 11 Strategy Update (Round 10 Results)

| Feature | Details |
| :--- | :--- |
| **Strategy Name** | **Cluster-Based Trust Regions (Local Penalisation)** |
| **Core Concept** | **Module 22 "Clustering Lens":** We treat the best historical point ($x_{best}$) as the centroid of a high-performing cluster. We assume the function is smooth enough that better points are likely "neighbors" within this cluster, separating "signal" from "noise". |
| **Acquisition Function** | **Expected Improvement (EI)** within **Trust Regions**. |
| **GP Kernel** | **Matern ($\nu=2.5$) + WhiteKernel**. We keep the robust kernel because we are zooming in, and Matern handles local irregularities better than RBF. |
| **Exploration ($\xi$)** | **0.001 (Pure Exploitation)**. We are no longer exploring globally. Within the cluster, we want the absolute best point. |
| **Optimization Method** | **L-BFGS-B with Dynamic Bounds**. <br>For each function, we define a "Local Cluster Box": $[x_{best} - R, x_{best} + R]$. <br> **Radius ($R$):** <br> - **0.05 (Tight)** for F5, F8 (Boundary Clusters). <br> - **0.15 (Moderate)** for F4, F6 (Complex Internal Clusters). <br> - **0.3 (Loose)** for F1, F2, F3, F7 (Scattered/Uncertain). |