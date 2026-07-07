# Python Bug Pattern Classifier

An ML experimentation project that classifies bug patterns from buggy and fixed Python code pairs.

## Goal

The goal of this project is to train and evaluate a supervised machine learning model. 

The final system will take buggy and fixed Python code as input and return:

- predicted bug pattern
- confidence score
- top-k alternative predictions
- short explanation of the predicted bug type

## Dataset

Primary Dataset: RunBugRun

Target Examples Include:

- buggy Python code
- fixed Python code
- bug pattern label
- optional test/result metadata

## Architecture

1. Data Loading:
- Load raw dataset files
- Filter examples to only Python code

2. Preprocessing:
- Extract buggy code, fixed code, and labels.
- Remove incomplete examples.
- Normalize labels.
- Create train/validation/test splits

3. Feature Engineering:
- Convert code into numerical features using TF-IDF.
- Experiment with buggy code, fixed code, and code diff representations.

4. Model Training:
- Train baseline classifiers such as Logistic Regression and Linear SVM.
- Handle label imbalance when needed.

5. Evaluation:
- Evaluate with accuracy, macro F1, weighted F1, top-k accuracy, and confusion matrix.
- Analyze which bug labels are commonly confused.

6.  Inference API:
- Save the trained model and vectorizer.
- Load them in a FastAPI application.
- Return predicted bug type, confidence score, and top-k predictions.

## MVP

The first version of this project will use:
- TF-IDF feature extraction
- Logistic Regression classifier
- bug pattern classification
- FastAPI inference endpoint

## Planned Tech Stack
- Python
- pandas
- scikit-learn
- TF-IDF
- Logistic Regression
- Linear SVM
- FastAPI
- Pytest
- Docker