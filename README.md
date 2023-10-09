# Deep Survival Analysis for Healthcare: An Empirical Study on Post-Processing Techniques

Survival analysis is a crucial tool in healthcare, allowing us to understand and predict time-to-event occurrences
using statistical and machine-learning techniques. As deep learning gains traction in this domain, a specific challenge
emerges: **neural network-based survival models** often produce discrete-time outputs, with the number of discretization
points being much fewer than the unique time points in the dataset, leading to potentially inaccurate survival
functions. To this end, our study explores post-processing techniques for survival functions. Specifically,
**interpolation and smoothing** can act as effective regularization, enhancing performance metrics integrated over time,
such as the Integrated Brier Score and the Cumulative Area-Under-the-Curve. We employed various regularization
techniques on diverse real-world healthcare datasets to validate this claim. Empirical results suggest a significant
performance improvement when using these post-processing techniques, underscoring their potential as a robust
enhancement for neural network-based survival models. These findings suggest that integrating the strengths of neural
networks with the non-discrete nature of survival tasks can yield more accurate and reliable survival predictions in
clinical scenarios.

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/archettialberto/interpolation_for_deep_survival_analysis
```

Start a Poetry shell:

```bash
poetry shell
```

Install the dependencies:

```bash
poetry install
```

## 🛠️ Usage

Run ```exps.py``` to start the experiments:

```bash
python exps.py
```

The results will be saved in the ```results``` directory.

## 📕 Bibtex Citation

```
TODO
```
