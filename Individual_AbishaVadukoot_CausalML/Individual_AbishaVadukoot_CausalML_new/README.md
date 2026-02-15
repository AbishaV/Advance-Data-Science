# What If? - Estimating Causal Effects with Machine Learning

A comprehensive Jupyter notebook exploring causal inference in machine learning, from foundational theory to hands-on implementation with real-world datasets.

## What This Project Covers

**Theory:** Potential outcomes framework, structural causal models, DAGs, confounders, colliders, mediators, and the backdoor criterion. All explained through a running coffee-productivity example.

**Data Preparation:** Why data prep in causal ML differs from predictive ML, covering missing data (MCAR/MAR/MNAR), causal feature selection, and encoding pitfalls.

**Worked Example 1 - LaLonde Job Training Dataset:**
Asks *does a job training program increase earnings?* The naive data says the program harmed participants by $15,000. Using propensity score matching, I recovered the true effect: a $1,772 gain, within $22 of the experimental benchmark.

**Worked Example 2 - IHDP Early Childhood Intervention:**
Asks *does an intervention improve cognitive scores, and who benefits most?* Using Double Machine Learning and Causal Forests, I found that the most vulnerable children (lower birth weight, younger mothers) benefit the most from the intervention.

## Datasets

| Dataset | Source | Size | Purpose |
|---------|--------|------|---------|
| LaLonde (Dehejia-Wahba) | [NBER](https://users.nber.org/~rdehejia/nswdata2.html) | 2,675 observations | Bias correction via propensity scores |
| IHDP | [CEVAE Repository](https://github.com/AMLab-Amsterdam/CEVAE/tree/master/datasets/IHDP/csv) | 747 observations | Heterogeneous treatment effects via Causal Forests |

## Methods Used

- Propensity Score Matching (Nearest Neighbor)
- Inverse Probability Weighting (IPW)
- Doubly Robust Estimation
- Double Machine Learning (Chernozhukov et al., 2018)
- Causal Forests (Athey and Imbens, 2016)
- DoWhy 4-step workflow (Model, Identify, Estimate, Refute)

## Installation

```
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels dowhy econml networkx scipy
```

If you hit NumPy compatibility issues, use a fresh virtual environment:

```
python -m venv causal_env
causal_env\Scripts\activate        # Windows
source causal_env/bin/activate     # Mac/Linux
pip install numpy==2.0.0 scipy==1.15.3 pandas matplotlib seaborn scikit-learn statsmodels dowhy econml networkx
```

## Project Structure

```
README.md
causal_inference_notebook.ipynb    # Main notebook (theory + code + examples)
quiz_questions.md                  # 15 MCQs on causal inference
psid_controls                      # Control dataset for Example 1: Lalonde
nswre74_treated                    # Treated dataset for Example 1: Lalonde
ihdp_npci_1.csv                    # Dataset for Example 2: IHDP
video_presentation                 # Script for the video walkthrough
```

## Key Results

**LaLonde:**

| Method | Estimate | Benchmark |
|--------|----------|-----------|
| Naive Comparison | -$15,205 | $1,794 |
| Propensity Score Matching | $1,772 | $1,794 |

**IHDP:**

| Subgroup | Estimated CATE | True ITE |
|----------|---------------|----------|
| Low Benefit | 3.128 | 3.255 |
| High Benefit | 4.229 | 4.445 |

## License

MIT License. See the notebook for full details.
