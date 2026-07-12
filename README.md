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

## Dataset Decisions

## Single-lable Filtering

The original RunBugRun dataset from Hugging Face. For the V1 baseline, I used the training split to generate diffs from buggy_code and fixed_code.

The original dataset had 133,705 examples. However, after filtering only to single labels, 35,962 labels were leftover. After examining the distribution of labels, I found that there were some examples with only a handful of examples (< 200) and decided to remove them.

The classes removed may be revisited in a future version.

I considered combining the rare clasess into one, but I decided to filter through these examples as they do not have a coherent bucket other than being "rare." Essentially, these are only grouped by frequency rather than semantic meaning.