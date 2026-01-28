## Overview
This project tackles a fundamental problem in Machine Learning: how to find the best settings (parameters) for a system when you don't know how the system works and testing it is expensive.

Over the course of 13 weeks, I optimized 8 synthetic "Black-Box" functions ranging from 2 dimensions to 8 dimensions. Unlike standard optimization where gradients are known, this challenge required balancing **Exploration** (gathering information about unknown areas) and **Exploitation** (refining the best-known results). My approach evolved from naive search methods to a sophisticated **Gaussian Process** model that explicitly accounts for data noise ("aleatory uncertainty"), culminating in a **Trust Region** strategy to pinpoint global maxima.

## Key Results (Week 13)
The final evaluation revealed distinct behaviors across the functions, highlighting the success of the noise-robust strategy:

| Function | Dim | Final Score | Strategy Outcome |
| :--- | :--- | :--- | :--- |
| **F5** | 4D | **8662.4** | **Solved.** Successfully identified the deterministic global basin. |
| **F1** | 2D | **0.1744** | **Converged.** Stable convergence to the global maximum. |
| **F4** | 4D | **0.4940** | **Stable.** Consistent performance in a multi-modal landscape. |
| **F8** | 8D | **3.64** | **High Stochasticity.** Historical best was ~9.99. The regression in the final round confirmed this function simulates high-variance noise (e.g., LLM temperature), making single-point estimates unreliable. |

## Strategy Evolution
The optimization strategy was not static; it adapted based on the "posterior" knowledge gained each week.

### Phase 1: Exploration (Weeks 1-3)
*   **Method:** Random Search & Standard Gaussian Process (RBF Kernel).
*   **Goal:** Establish a baseline and map the rough topology of the search space.
*   **Insight:** Standard kernels failed to model the "jagged" nature of high-dimensional functions (F6-F8).

### Phase 2: Robust Noise Modeling (Weeks 4-10)
*   **Method:** **Matern 2.5 Kernel + WhiteKernel**.
*   **Goal:** Handle the "stochastic" nature of the functions.
*   **Insight:** The functions simulate noisy processes (like LLM decoding). Adding a `WhiteKernel` allowed the model to treat minor score fluctuations as noise rather than structural features, preventing overfitting to outliers.

### Phase 3: Trust Regions & Dimensionality Reduction (Weeks 11-12)
*   **Method:** **Cluster-Based Trust Regions**.
*   **Goal:** Combat the "Curse of Dimensionality."
*   **Insight:** Instead of searching the entire 8D hypercube, I restricted the optimizer to a hypersphere radius $R$ centered on the best historical point. This "Local Penalisation" forced the model to refine the gradient of the most promising peak.

### Phase 4: The Finite Horizon Pivot (Week 13)
*   **Method:** **Experience Replay**.
*   **Goal:** Verification.
*   **Insight:** In the final round, the value of new information drops to zero. I switched from predictive modeling to a deterministic policy, re-querying historical maximums to separate "lucky" outliers from stable optima.

## Repository Structure

*   `capstone_week13.ipynb`: The main notebook containing the optimization loop, GPR model definitions, and acquisition functions.
*   `bbo_master_w13.csv`: The **Final Dataset** containing 176 evaluations (22 per function).
*   `add_data/`: Contains weekly JSON logs of inputs and outputs.
*   `create_bbo_master_w13.py`: Utility script for aggregating weekly results into the master dataset.

## Documentation
For detailed technical analysis and transparency reports:

*   **[Project Datasheet](./DATASHEET.md)**: Detailed breakdown of the dataset composition, provenance, and preprocessing steps (rounding/scaling).
*   **[Model Card](./MODEL_CARD.md)**: Technical specifications of the Gaussian Process architecture, kernel choices, and limitation analysis.

## Dependencies
The project relies on the standard Python Data Science stack:
*   `scikit-learn` (Gaussian Processes, PCA)
*   `numpy` / `pandas` (Data manipulation)
*   `scipy` (L-BFGS-B Optimization)

## Contact
**Project Author:** Larry Lai
*This project was completed as part of the Imperial College Business School ML/AI Professional Certificate.*
