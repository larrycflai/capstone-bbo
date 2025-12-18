# Datasheet for Black-Box Optimisation (BBO) Capstone Data Set

## 1. Motivation

*   **Why was this data set created?**
    This data set was created as part of the **Black-Box Optimisation (BBO) Capstone Challenge**. The primary objective was to maximise eight unknown, synthetic "black-box" functions that simulate real-world machine learning tasks (e.g., radiation detection, robot control, drug discovery) where evaluations are expensive or limited.

*   **What task does it support?**
    The data supports the task of **Bayesian Optimisation**. It records the iterative search process (inputs $X$ and outputs $Y$) used to map and maximise functions with varying dimensionality (2D to 8D) under constraints of limited query budgets.

## 2. Composition

*   **What does the instances represent?**
    Each row in the dataset represents a single function evaluation. It consists of:
    *   **Function ID:** Unique identifier (1–8).
    *   **Round:** The iteration number (0 for initial data, 1–10 for weekly submissions).
    *   **Inputs ($X_1 \dots X_8$):** The coordinates queried within the search space.
    *   **Output ($Y$):** The scalar score returned by the black-box function.

*   **How many instances are there?**
    The dataset contains **152 total data points** (as of Round 10), distributed as:
    *   **8 Functions** independent of each other.
    *   **19 evaluations per function** (10 initial random points + 9 weekly strategic queries).

*   **What is the format of the data?**
    *   **Master File:** `bbo_master_w10.csv` (Aggregated CSV containing all history).
    *   **Weekly Queries:** JSON files (e.g., `week10_clean_inputs.json`) containing specific weekly submissions.

*   **Does the data set contain all possible instances or is it a sample?**
    It is a highly sparse sample. The search space is continuous, discretised to 6 decimal places. With only 19 points covering an 8-dimensional hypercube (Function 8), the data is extremely sparse, leaving vast regions of the search space unexplored.

*   **Are there any known errors, sources of noise, or redundancies?**
    *  **Noise:** The objective functions exhibit **observational noise** (stochastically). Repeated evaluations or evaluations in close proximity display variance that cannot be explained by smooth mathematical functions alone.
    *  **Mitigation:** This noise required the introduction of a **WhiteKernel** (noise_level=0.1) in the Gaussian Process model to distinguish between signal and variance.
    *  **Context:** This project interacts with the functions via continuous numerical inputs. Therefore, the noise is treated mathematically as aleatoric uncertainty rather than prompt instability.

## 3. Collection Process

*   **How was the data associated with each instance acquired?**
    Data was acquired iteratively over a 10-week period. Each week, a machine learning strategy proposed new $X$ coordinates, which were submitted to the capstone portal to retrieve the corresponding $Y$ values.

*   **What mechanisms or strategies were used to collect the data?**
    The query generation strategy evolved to address the "black-box" nature of the functions:
    *   **Weeks 1–2 (Exploration):** Used **Upper Confidence Bound (UCB)** and Adaptive Mix strategies to identify promising regions.
    *   **Weeks 3–4 (Smoothness Assumption):** Experimented with **RBF Kernels**, assuming the functions were smooth (later disproven by jagged results).
    *   **Weeks 5–8 (Robust Exploitation):** Switched to **Matern Kernels** with **WhiteKernel** noise modeling to handle the jagged, noisy LLM outputs. Used **Expected Improvement (EI)** to refine peaks.
    *   **Week 9 (Global Restart):** Utilized **Random Search** to escape local optima and reduce global variance.
    *   **Week 10 (Posterior Refinement):** Reverted to Bayesian Optimization (EI) to capitalize on the variance reduction from Week 9.

*   **Over what timeframe was the data collected?**
    The data was collected weekly from the start of the Stage 2 Capstone through Week 10.

## 4. Preprocessing and Uses

*   **Was any preprocessing/cleaning done?**
    *   **Scaling:** Input data ($X$) was standardised using `StandardScaler` to ensure numerical stability for Gaussian Processes regression.
    *   **Rounding:** All query inputs were strictly rounded to **6 decimal places** to meet the submission requirements.
    *   **Dimensionality Enforcement:** Queries were strictly truncated to their specific dimensions (e.g., 2D, 8D) to ensure no "ghost" dimensions influenced the result.

*   **What are the intended uses of the data set?**
    *   Benchmarking Bayesian Optimisation strategies (specifically Guassian Process) on high-dimensional, expensive-to-evaluate functions.
    *   Analyzing the trade-off between **Exploration** (UCB) and **Exploitation** (EI) in sparse data environments.

*   **What are the inappropriate uses?**
    *   **Supervised Learning:** It is inappropriate to use this dataset to train complex supervised models (like Deep Neural Networks). With only ~19 data points covering an 8-dimensional hypercube, the data is far too sparse for valid generalisation.
    *   **Deterministic Ground Truth:** Users should not treat the function outputs ($Y$) as noiseless ground truth. The functions simulate real-world tasks (e.g., radiation detection) that contain inherent **observational noise**, necessitating the use of noise-handling kernels (like **WhiteKernel**) rather than perfect interpolation.
    *   **Extrapolation:** It is inappropriate to extrapolate trends outside the bounded search of the unit hypercube $[1]^d$, where $d$ ∈ {2, 3, 4, 5, 6, 7, 8} corresponds to the specific function's dimensionality.

## 5. Distribution and Maintenance

*   **How will the data be distributed?**
    The data is hosted publicly on this GitHub repository.

*   **Will the data be updated?**
    Yes, the dataset is updated weekly with one new data point per function until the conclusion of the Capstone project (Week 13).

*   **Who is responsible for maintaining the data set?**
    The student researcher maintaining this repository.
