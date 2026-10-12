# Classification-Algorithms-Model-Building
Crowdfunding Campaign Success Prediction

A beginner data science and machine learning portfolio project that
explores crowdfunding campaign data and builds classification models to
predict whether a campaign is successful.

Project Overview

The goal of this project is to practise the end-to-end machine learning
workflow: exploring a dataset, preprocessing features, training
classification algorithms, evaluating model performance, and tuning
model hyperparameters.

Five classification algorithms are compared:

Random Forest

K-Nearest Neighbors (KNN)

Decision Tree

Gaussian Naive Bayes

Support Vector Machine (SVM)

The notebook evaluates models using accuracy, precision, recall,
F1-score, confusion matrices, training-versus-testing performance, and
hyperparameter tuning with GridSearchCV.

Portfolio note: The metrics below are from the model comparison
recorded in this project notebook. Results can change when
preprocessing, random splits, or hyperparameters change.

Objectives

Inspect and understand the crowdfunding dataset.

Check for missing values and duplicate records.

Encode categorical variables for machine learning.

Select informative features.

Split data into training and testing sets.

Scale features where appropriate.

Train and compare five classification models.

Evaluate models using multiple classification metrics.

Use cross-validation and hyperparameter tuning.

Identify a preferred model and discuss limitations and improvements.

Dataset

The notebook loads the crowdfunding dataset from the following source:

https://raw.githubusercontent.com/GeethaGunasekaran1/Dataset_rep/refs/heads/main/crowdfunding.csv

The dataset contains campaign-related fields such as:

CampaignID

GoalAmount

RaisedAmount

DurationDays

NumBackers

Category

LaunchMonth

Country

Currency

OwnerExperience

VideoIncluded

SocialMediaPresence

NumUpdates

IsSuccessful --- target variable

The target variable is IsSuccessful, which represents the campaign
outcome.

The notebook removes the Unnamed: 0 column and excludes CampaignID
from the model features because it is an identifier rather than a
meaningful predictive measurement. It also checks missing values and
duplicate rows.

Tools and Technologies

Python

Jupyter Notebook

pandas

NumPy

Matplotlib

Seaborn

scikit-learn

Project Workflow

1. Data exploration and preprocessing

Load and inspect the dataset.

Check missing values and duplicate rows.

Remove the unnecessary index column.

Identify numerical and categorical features.

Apply one-hot encoding to categorical features.

Separate the target (IsSuccessful) from the input features.

2. Feature selection and scaling

Use SelectKBest with f_classif to select the 10 highest-scoring
features.

Apply StandardScaler for models that are sensitive to feature
scales, particularly KNN and SVM.

Split the data into training and testing sets using an 80:20 split
and stratification.

3. Baseline model building

Train and evaluate the following classifiers:

Algorithm                           Purpose

Random Forest                       Combines multiple decision trees to
make predictions.

K-Nearest Neighbors (KNN)           Predicts a class using nearby
observations.

Decision Tree                       Learns decision rules from the
features.

Gaussian Naive Bayes                Applies Bayes' theorem with a
conditional-independence
assumption.

4. Model evaluation

The notebook compares training and testing results using:

Accuracy: Overall proportion of correct predictions.

Precision: Proportion of predicted successful campaigns that
were actually successful.

Recall: Proportion of actually successful campaigns identified
by the model.

F1-score: Harmonic mean of precision and recall.

Confusion matrix: Counts true positives, true negatives, false
positives, and false negatives.

Accuracy gap: Training accuracy minus testing accuracy; a large
positive gap can indicate overfitting.

5. Cross-validation and hyperparameter tuning

GridSearchCV is used to explore selected hyperparameters for the
classifiers. The searches use F1-score as the optimization metric and
cross-validation to compare parameter combinations. SVM tuning is
performed on a training subset to reduce computational cost.

6. Final model comparison

The tuned models are compared using their testing metrics. In the
recorded final comparison, Naive Bayes was selected based on its test
F1-score.

Results

The following table summarizes the final model comparison recorded in
the project:

Model           Train         Test         Test  Test Recall         Test Accuracy Gap
Accuracy     Accuracy    Precision                  F1-score

Naive          66.50%       67.02%       74.46%       52.01%   61.24%        -0.53
Bayes                                                                       percentage
points

KNN            79.73%       61.05%       61.84%   58.11%       59.92%        18.68
percentage
points

Random         77.68%       65.14%       71.57%       50.47%       59.20%        12.54
Forest                                                                      percentage
points

Decision       67.02%   67.30%       80.00%       46.30%       58.65%        -0.28
Tree                                                                        percentage
points

Key findings

Naive Bayes achieved the highest test F1-score (61.24%) and was
selected as the preferred model using F1-score as the main
criterion.

Decision Tree achieved the highest test accuracy (67.30%).

SVM achieved the highest test precision (80.67%).

KNN achieved the highest test recall (58.11%), but its large
accuracy gap suggests possible overfitting.

Random Forest also showed a noticeable training--testing
accuracy gap.

Recall remains an area for improvement across the models, so a
considerable number of successful campaigns may still be missed.

Limitations and Future Improvements

Improve recall: Evaluate classification thresholds and the
precision--recall trade-off.

Reduce overfitting: Tune model complexity and validate using
cross-validation.

Improve feature engineering: Investigate additional useful
features and remove irrelevant ones.

Prevent preprocessing leakage: Put feature selection and scaling
inside a scikit-learn pipeline so each cross-validation fold learns
preprocessing only from its training portion.

Check target leakage: If the intended use is to predict success
before a campaign launches, RaisedAmount and NumBackers may not
be available at prediction time. Exclude them in that scenario and
evaluate the model again.

Validate the final model: Compare cross-validation scores and
held-out test metrics before deployment.

Explore additional approaches: Consider other classifiers and
calibration methods if justified by the project goal.

How to Run the Project

Clone or download this repository.

Install the required libraries.

Open Classification_Algorithms_Model_Building.ipynb in Jupyter
Notebook or Visual Studio Code.

Run the notebook cells in order.

Install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn jupyter

The notebook reads the dataset from its online CSV source, so an
internet connection is required unless you download the CSV and update
the dataset path.

Repository Structure

crowdfunding-success-prediction/
├── Classification_Algorithms_Model_Building.ipynb
└── README.md

You can add a requirements.txt, exported charts, or a local copy of
the dataset if needed. Avoid committing sensitive information or large
files unnecessarily.

What I Learned

Through this project, I practised:

Data inspection and preprocessing

Categorical encoding and feature selection

Training and testing data preparation

Feature scaling

Classification model implementation

Model evaluation using multiple metrics

Confusion matrix interpretation

Cross-validation and hyperparameter tuning

Comparing models and explaining their limitations

Conclusion

This project provided practical experience with a complete
classification workflow, from data preprocessing to model comparison.
Based on the recorded test F1-scores, Naive Bayes was selected as the
preferred model, while the results also highlighted opportunities to
improve recall and address possible overfitting. The project reinforced
the importance of selecting models according to the problem objective
and evaluating them with more than accuracy alone.

Project type: Supervised Machine Learning --- Binary Classification
Status: Learning portfolio project
Author: Aspiring Data Scientist
