# Python Bug Pattern Classifier

An ML experimentation project that classifies bug patterns from buggy and fixed Python code pairs.

## Goal

The goal of this project is to train and evaluate a supervised machine learning model.

The final system will take buggy and fixed Python code as input and return:

- Predicted bug pattern

---

## Dataset

**Primary Dataset:** RunBugRun

**Target Examples Include:**

- Buggy Python code
- Fixed Python code
- Bug pattern label

---

## Architecture (V1)

### 1. Data Loading

- Load raw dataset files
- Filter examples to only Python code (already done by RunBugRun)

### 2. Preprocessing

- Extract buggy code, fixed code, and labels
- Remove incomplete examples
- Normalize labels
- Create train/test splits

### 3. Feature Engineering

- Convert code into numerical features using TF-IDF
- Experiment with buggy code, fixed code, and code diff representations

### 4. Model Training

- Train baseline classifiers:
  - Logistic Regression
  - Linear SVM

### 5. Evaluation

- Evaluate with:
  - Accuracy
  - Macro F1
  - Weighted F1
  - Top-k Accuracy
  - Confusion Matrix
- Analyze which bug labels are commonly confused

### 6. Save the trained model and vectorizer

### 7. Inference API (V2/V3)

- Load them in a FastAPI application
- Return:
  - Predicted bug type
  - Confidence score
  - Top-k predictions

---

## Tech Stack

- Python
- pandas
- scikit-learn
- TF-IDF
- Logistic Regression
- Linear SVM

---

## Dataset Decisions

### Single-Label Filtering

The original RunBugRun dataset comes from Hugging Face. For the V1 baseline, I used the training split to generate diffs from `buggy_code` and `fixed_code`.

The original dataset had **133,705** examples. However, after filtering to only single-label examples, **35,926** remained.

After examining the distribution of labels, I found that some classes contained only a handful of examples (<200) and decided to remove them. This left **35,641** examples for the V1 baseline.

The removed classes may be revisited in a future version.

I considered combining the rare classes into a single category. However, I decided against this because they do not represent a coherent programming concept—they are only grouped together because they occur infrequently.

---

## V1 Metrics

### Baseline Comparison

Logistic Regression slightly outperformed the Linear SVM.

There was confusion between the Assignment and Call classes. The Expression class performed the worst and was most frequently confused with Call.

This suggests that the choice of classifier may not be the primary reason for these errors. I will continue with V2 to determine whether TF-IDF is the limiting factor while also performing additional error analysis to better understand these mistakes.

### Logistic Regression

- Overall Accuracy: **70.81%**
- Assignment: **75.14%** recall, some confusion with the Call class
- Call: **89.08%** recall
- Control Flow: **86.28%** recall
- Expression: **21.44%** recall (lowest-performing class, often predicted as Call)
- Identifier: **23.66%** recall (often predicted as Call)

### Linear SVM

- Overall Accuracy: **69.97%**
- Assignment: **65.32%** recall, some confusion with the Call class
- Call: **89.63%** recall
- Control Flow: **87.49%** recall
- Expression: **21.17%** recall (lowest-performing class, often predicted as Call)
- Identifier: **24.53%** recall (often predicted as Call)

---

## Deeper Error Analysis

After completing the error analysis (more detailed evidence can be found in [`03_error_analysis.ipynb`](notebooks/03_error_analysis.ipynb)), I found several limitations of the TF-IDF classifier.

For example, the classifier performed well on labels that were strongly associated with common Python keywords. The `control_flow` class performed well because it contained tokens such as `if`, `elif`, `break`, `or`, and `while`, which were clear indicators of the class.

The TF-IDF classifier struggled on classes such as `expression` and `identifier`, where there were no clear token-level indicators of the programming concept. Properly classifying these examples requires understanding the surrounding context rather than only the individual tokens contained in the diff.

These limitations are the primary motivation for a V2 of this project, where I will use PyTorch and more modern machine learning techniques to improve metrics such as accuracy, precision, and recall.