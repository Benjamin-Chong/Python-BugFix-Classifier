# Python Bug Pattern Classifier

An ML experimentation project that classifies bug patterns from buggy and fixed Python code pairs.

## Goal

The goal of this project is to train and evaluate a supervised machine learning model. 

The final system will take buggy and fixed Python code as input and return:

- predicted bug pattern

## Dataset

Primary Dataset: RunBugRun

Target Examples Include:

- buggy Python code
- fixed Python code
- bug pattern label

## Architecture V1

1. Data Loading:
- Load raw dataset files
- Filter examples to only Python code (already done by RunBugRun)

2. Preprocessing:
- Extract buggy code, fixed code, and labels.
- Remove incomplete examples.
- Normalize labels.
- Create train/validation/test splits.

3. Feature Engineering:
- Convert code into numerical features using TF-IDF.
- Experiment with buggy code, fixed code, and code diff representations.

4. Model Training:
- Train baseline classifiers: Logistic Regression and Linear SVM.

5. Evaluation:
- Evaluate with accuracy, macro F1, weighted F1, top-k accuracy, and confusion matrix.
- Analyze which bug labels are commonly confused.

6.  Inference API (V2/3):
- Save the trained model and vectorizer.
- Load them in a FastAPI application.
- Return the predicted bug type, confidence score, and top-k predictions.

## MVP

The first version of this project will use:
- TF-IDF feature extraction
- Logistic Regression classifier
- bug pattern classification

## Planned Tech Stack
- Python
- pandas
- scikit-learn
- TF-IDF
- Logistic Regression
- Linear SVM
- Pytest

## Dataset Decisions

## Single-label Filtering

The original RunBugRun dataset comes from Hugging Face. For the V1 baseline, I used the training split to generate diffs from buggy_code and fixed_code.

The original dataset had 133,705 examples. However, after filtering only to single labels, 35,926 labels remained. After examining the distribution of labels, I found that there were some classes with only a handful of examples (< 200) and decided to remove them. This left me with 35,641 examples for the V1 baseline.

The classes removed may be revisited in a future version.

I considered combining the rare classes into one, but I decided to filter through these examples as they do not have a coherent bucket other than being "rare." Essentially, these are only grouped by frequency rather than semantic meaning.

## V1 Metrics

### Baseline Comparison

Logistic Regression slightly outperformed the Linear SVM. There was confusion between the Assignment class and the call class. The expression class was the worst class and was confused with call. This suggests that the choice of classifier may not the primary reason for these errors. I will continue with V2 to figure out if TF-IDF is the limiting factor and I will also continue error analysis to figure out why.

### Notable Logistic Regression Findings
- Overall Accuracy: 70.81%
- Assignment: 75.14% recall, some confusion with the call class
- Call: 89.08% recall
- Control-Flow: 86.28% recall
- Expression: 21.44% recall, worst class and the model predicted call rather than expression
- Identifier: 23.66% recall, model also predicted call rather than identifier

### Notable Linear SVC Findings
- Overall Accuracy: 69.97%
- Assignment: 65.32% recall, some confusion with the call class (lower than Logistic Regression)
- Call: 89.63% recall, slightly better than Logistic Regression
- Control-Flow: 87.49% recall, slightly better than Logistic Regression
- Expression: 21.17% recall, worst class and the model predicted call rather than expression (lower than Logistic Regression)
- Identifier: 24.53% recall, model also predicted call rather than identifier (higher than Logistic Regression)