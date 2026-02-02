# Logistic-Comparison

________________________________________
Logistic Regression: Scratch vs Scikit-learn

This project implements Logistic Regression from scratch using NumPy and compares it against Scikit-learn’s LogisticRegression on a synthetic binary classification dataset.
The goal is to understand the internal mechanics of logistic regression while validating correctness against a well-tested library implementation.
________________________________________
Project Structure

├── logreg_model_comparison.py

└── README.md
________________________________________
Overview

The project covers the entire machine learning workflow:

•	Synthetic dataset generation

•	Feature scaling

•	Logistic Regression implementation from scratch

•	Training using batch gradient descent

•	Model evaluation and comparison

•	Visualization of:

o	Training loss curve

o	Probability predictions

o	Test data projection
________________________________________
 Dataset

•	Generated using sklearn.datasets.make_classification

•	500 samples

•	8 features

•	Binary classification

•	5 informative features

•	Train/Test split: 75% / 25%

•	Features are standardized using training-set statistics
________________________________________
Scratch Logistic Regression

The custom implementation includes:

•	Sigmoid activation

•	Log loss (binary cross-entropy)

•	Batch gradient descent

•	Learnable weights and bias

•	Probability and class prediction methods

•	Loss tracking for visualization

Key Components

sigmoid(z)

log_loss(y_true, y_pred)

fit(X, y)

predict_proba(X)

predict(X)

Training is performed using batch gradient descent with configurable learning rate and number of iterations.
________________________________________
Scikit-learn Comparison

The scratch model is compared against:

sklearn.linear_model.LogisticRegression

•	Solver: lbfgs

•	No regularization (penalty=None)

•	Same scaled features and training data
________________________________________
Evaluation Metrics

Both models are evaluated on the test set using:

•	Accuracy

•	Precision

•	Recall

•	F1 Score

This confirms that the scratch implementation performs comparably to Scikit-learn’s optimized version.
________________________________________
Visualizations

The script produces the following plots:

1. Training Loss Curve

Shows log loss decreasing over iterations for the scratch model.

2. Probability Prediction Comparison

Scatter plot comparing predicted probabilities from:

•	Scratch model

•	Scikit-learn model

(Points align closely along the diagonal)

3.Test Data Projection

2D visualization of test samples (first two features) colored by true labels.
________________________________________
 How to Run

Requirements

pip install numpy matplotlib scikit-learn

Run the Script

python logreg_model_comparison.py
________________________________________
Key Takeaways

•	Logistic Regression can be implemented with relatively simple math and NumPy

•	Feature scaling is critical for gradient descent convergence

•	A well-implemented scratch model can closely match Scikit-learn’s output

•	Visualizations help validate training behavior and prediction consistency
________________________________________
License

This project is open for educational and learning purposes.
________________________________________
Conclusion

This project shows how Logistic Regression can be implemented from scratch using NumPy and still perform similarly to Scikit-learn’s built-in model. By building the model step by step, we gain a clearer understanding of how training, prediction, and optimization work.

Overall, this comparison helps beginners move from simply using machine learning libraries to understanding what happens behind the scenes.
