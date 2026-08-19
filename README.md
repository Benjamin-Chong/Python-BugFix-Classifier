# Python Bug Pattern Classifier

An ML experimentation project that classifies bug patterns from buggy and fixed Python code pairs.

## Goal

The goal of this project is to build and compare progressively more capable supervised machine learning models for classifying Python bug-fix patterns.

Given the difference between buggy and fixed Python code, the model predicts one of five classes:

* `assignment`
* `call`
* `control_flow`
* `expression`
* `identifier`

The project currently contains three completed versions:

* **V1:** TF-IDF with classical machine learning classifiers
* **V2:** Learned token embeddings with masked mean pooling in PyTorch
* **V3:** Sequence-aware Transformer encoder with positional encoding and self-attention in PyTorch.

---

## Dataset

**Primary Dataset:** RunBugRun, accessed through Hugging Face

Each example contains:

* Buggy Python code
* Fixed Python code
* One or more bug-pattern labels

A diff is generated from each buggy and fixed code pair. The models use this diff as their input.

### Dataset Filtering

The original RunBugRun training split contained **133,705** examples.

I limited the scope of this project to single-label classification. After removing examples without labels and examples containing multiple labels, **35,926** examples remained.

I also removed the following classes because they contained fewer than 200 examples:

* `literal`
* `function`
* `variable_access`
* `io`
* `try_catch`

This left **35,641** examples across the five target classes.

I considered combining the rare classes into one category. However, I decided against this because the classes do not represent one coherent programming concept. They would only be grouped together because they occur infrequently.

---

## V1: TF-IDF Baseline

### Architecture

```text
Buggy and fixed code
→ generated diff
→ TF-IDF representation
→ classical classifier
→ predicted bug pattern
```

V1 compared two classifiers:

* Logistic Regression
* Linear SVM

Both models used TF-IDF representations of the generated diffs.

### Original V1 Evaluation

The original cleaned dataset was divided into:

* **28,512 training examples**
* **7,129 test examples**

Logistic Regression slightly outperformed the Linear SVM.

#### Logistic Regression

* Accuracy: **70.81%**
* Assignment recall: **75.14%**
* Call recall: **89.08%**
* Control-flow recall: **86.28%**
* Expression recall: **21.44%**
* Identifier recall: **23.66%**

#### Linear SVM

* Accuracy: **69.97%**
* Assignment recall: **65.32%**
* Call recall: **89.63%**
* Control-flow recall: **87.49%**
* Expression recall: **21.17%**
* Identifier recall: **24.53%**

### V1 Error Analysis

The TF-IDF models performed well on classes associated with recognizable Python keywords.

For example, `control_flow` was strongly associated with tokens such as:

* `if`
* `elif`
* `break`
* `or`
* `while`

The models struggled with `expression` and `identifier`. These classes did not have the same clear token-level indicators and were frequently predicted as `call` or `assignment`.

The full V1 analysis is available in [`04_error_analysis.ipynb`](notebooks/v1/04_error_analysis.ipynb).

These results motivated V2, where I replaced the fixed TF-IDF representation with token embeddings learned during training.

---

## V2: PyTorch Embedding Classifier

### Development Split

V1 originally used a train-and-test workflow. For V2, I introduced a validation set so that vocabulary decisions, model selection, and error analysis could occur without using the held-out test set.

The original V1 training set was divided into:

* **22,809 V2 training examples**
* **5,703 V2 validation examples**

The split used a fixed random seed and stratification to preserve the class distribution.

The original test set was not used during V2 development.

### Architecture

```text
Generated diff
→ custom Python tokenizer
→ token IDs
→ dynamic batch padding
→ padding mask
→ 128-dimensional token embeddings
→ masked mean pooling
→ Linear(128, 5)
→ five class logits
```

V2 was implemented from scratch using PyTorch rather than using a pre-trained language model.

Each token is assigned a learned 128-dimensional embedding. The embeddings for the real tokens in each example are averaged using masked mean pooling. The resulting vector is passed into a linear classifier that produces one logit for each class.

### Tokenization

The custom tokenizer preserves information specific to code diffs:

* `<ADD>` represents an added line.
* `<DELETE>` represents a deleted line.
* `<INDENT_n>` represents the indentation level.
* Python comments are excluded.
* `<PAD>` represents dynamically added padding.
* `<UNK>` represents tokens outside the training vocabulary.

### Vocabulary

The vocabulary was constructed using only the V2 training data to prevent validation leakage.

Token frequency was measured using document frequency, where each document represents one tokenized diff.

I selected `min_freq=4`. Tokens appearing in fewer than four training examples were removed, except for:

* `<ADD>`
* `<DELETE>`
* Indentation tokens

The final vocabulary contained **947 entries**, including `<PAD>` and `<UNK>`.

Although the threshold removed **81.67%** of the unique training token types, the validation out-of-vocabulary rate remained low at **2.04%**. This showed that most of the removed tokens occurred very rarely.

### Sequence Length

I initially considered truncating sequences to reduce computation. After manually inspecting longer examples, I found that the change responsible for the label could appear at the beginning, middle, or end of the diff.

Because truncation could remove the actual fix, I kept every token.

Dynamic padding pads each batch only to the length of its longest sequence. Because the batches are randomly shuffled rather than grouped by length, some padding waste may remain. Length-aware batching could be considered later if profiling shows that padding is a meaningful performance bottleneck.

### Training Configuration

* Random seed: **42**
* Batch size: **32**
* Embedding dimension: **128**
* Optimizer: **Adam**
* Learning rate: **0.001**
* Epochs: **15**
* Loss function: **Cross-Entropy Loss**
* Checkpoint criterion: **Lowest validation loss**

The training function records training and validation loss and accuracy for every epoch. The checkpoint saves the model state from the epoch with the lowest validation loss.

---

## Fair V1 Versus V2 Validation Comparison

To create a fair comparison, I retrained the V1 Logistic Regression model using the same **22,809 V2 training examples** used by V2.

The TF-IDF vectorizer was fitted only on the V2 training data. Both models were then evaluated on the same **5,703 validation examples**.

| Model                           | Validation Accuracy | Macro F1 | Weighted F1 |
| ------------------------------- | ------------------: | -------: | ----------: |
| V1 TF-IDF + Logistic Regression |              0.7058 |   0.5891 |      0.6769 |
| V2 Embeddings + Mean Pooling    |              0.7780 |   0.7249 |      0.7748 |

Compared with the retrained V1 model, V2 improved:

* Accuracy by **7.22 percentage points**
* Macro F1 by **13.58 percentage points**
* Weighted F1 by **9.79 percentage points**

### Class-Level Results

V2 improved every class by F1 score, but the gains were not evenly distributed.

The largest improvements occurred in `expression` and `identifier`.

#### Expression

In V1:

* **50.6%** of expression examples were predicted as `call`.
* **22.6%** were predicted as `assignment`.
* Expression recall was **21.8%**.

In V2:

* Expression-to-call errors decreased to **28.5%**.
* Expression-to-assignment errors decreased to **10.7%**.
* Expression recall increased to **55.7%**.

#### Identifier

In V1:

* **41.0%** of identifier examples were predicted as `assignment`.
* **30.1%** were predicted as `call`.
* Identifier recall was **22.5%**.

In V2:

* Identifier-to-assignment errors decreased to **4.6%**.
* Identifier-to-call errors decreased to **21.4%**.
* Identifier recall increased to **52.0%**.

One tradeoff was that identifier-to-expression errors increased from **4.6%** in V1 to **20.2%** in V2. However, this occurred alongside a substantial increase in correct identifier predictions.

### Additional Tradeoffs

V2 did not improve every metric for every class:

* Assignment recall decreased from **75.6%** to **70.7%**.
* Call recall decreased from **88.5%** to **87.4%**.
* Assignment-to-expression errors increased from **6.1%** to **14.9%**.

Despite these recall decreases, assignment and call both achieved higher F1 scores because V2 made larger improvements to their precision.

---

## V2 Limitations and V3 Motivation

V2 produced substantial improvements over the TF-IDF baseline, especially for the two classes that V1 struggled with most.

However, `expression` and `identifier` remained the weakest classes.

V2 learns token representations, but masked mean pooling averages those representations into one vector. This means the model considers which tokens appear but does not preserve their order.

The remaining errors are consistent with the hypothesis that `expression` and `identifier` require relationships between tokens and surrounding code structure that mean pooling cannot represent. The current evidence does not prove that missing sequence context is the cause, but it gives V3 a clear hypothesis to test.

V3 introduces a sequence-aware architecture while preserving the same development split and evaluation methodology.

---

## V3: Sequence-Aware Tranformer Classifier

### Architecture
V3 extends V2 while preserving the same tokenizer, vocabulary, data splits, and 128-dimensional token embeddings. This controlled design allowed the experiment to focus on whether sequence-aware contextualization would improve classification performance.
V3 introduces the following architectural components:

1. Sinusoidal positional encoding with a maximum sequence length of `1024`
2. Two Transformer encoder layers
3. Four self-attention heads
4. A feed-forward dimension of `256`
5. Dropout of `0.1`
6. Padding-aware self-attention
7. Masked mean pooling after contextualization
8. A final `Linear(128, 5)` classifier

V2 mean-pools independent token embeddings, while V3 first contextualizes each token using its position and relationships with other tokens. It then mean-pools those contextualized representations before classification.

### Training Configuration
* Random seed: **42**
* Batch size: **32**
* Embedding dimension: **128**
* Optimizer: **Adam**
* Learning rate: **0.001**
* Epochs: **15**
* Loss function: **Cross-Entropy Loss**
* Checkpoint criterion: **Lowest validation loss**
* Selected checkpoint: **Epoch 10**
* Best validation loss: **0.3144**

### Validation Results
| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| V2 Embeddings + Mean Pooling | 0.7780 | 0.7249 | 0.7748 |
| V3 Transformer Encoder | 0.8773 | 0.8356 | 0.8748 |

V3 improved validation accuracy from **0.7780** to **0.8773**, macro F1 from **0.7249** to **0.8356**, and weighted F1 from **0.7748** to **0.8748**. The macro F1 improvement is especially meaningful because every class contributes equally to this metric, showing that V3’s gains were not driven only by the majority class. Expression F1 increased from **0.5786** to **0.7864**, while identifier F1 increased from **0.5941** to **0.7129**. These results support the hypothesis that the weaker classes benefited from sequence-aware contextualization. However, they do not prove that context alone caused the improvement because V3 also introduced additional model capacity through its Transformer encoder.


## Final Test Set Comparison

### Evaluation Protocol
All three models were frozen before the final evaluation and were evaluated on the same **7,129** held-out test examples. For the fair V1 comparison, the TF-IDF vectorizer was fitted and the Logistic Regression model was trained using only the **22,809** training examples. V2 and V3 used the same tokenizer, frozen vocabulary, numericalized test set, and DataLoader. The test set was used only for final evaluation and was not used for model selection or tuning.

### Overall Test Results
| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| V1 TF-IDF + Logistic Regression | 0.7068 | 0.5888 | 0.6755 |
| V2 Embeddings + Mean Pooling | 0.7725 | 0.7120 | 0.7687 |
| V3 Transformer Encoder | 0.8752 | 0.8384 | 0.8725 |

### Test F1 by Class
| Class | V1 | V2 | V3 |
|---|---:|---:|---:|
| Assignment | 0.6363 | 0.7348 | 0.8334 |
| Call | 0.8133 | 0.8401 | 0.9128 |
| Control Flow | 0.8673 | 0.8920 | 0.9439 |
| Expression | 0.2902 | 0.5703 | 0.7722 |
| Identifier | 0.3367 | 0.5225 | 0.7297 |

### Final Test Interpretation
V3 achieved the best overall test performance, reaching **87.52%** accuracy, **0.8384** macro F1, and **0.8725** weighted F1. It improved on V2 across every class, with the largest F1 gains occurring for expression and identifier. These were the classes that were expected to benefit most from preserving token order and learning relationships between tokens. V3’s test results were also very close to its validation results, showing that the improvement generalized to unseen test data. However, identifier recall remained at **0.6250**, showing that the model still has difficulty identifying some examples from the smallest class. Overall, the results strongly support the hypothesis that, under this experimental setup, a sequence-aware Transformer representation is more effective than TF-IDF or mean-pooled token embeddings for this classification task.

## Limitations and Future Work
- V3 added both sequence awareness and model capacity, so their individual effects were not isolated.
- The `identifier` class remained small, with only `216` test examples.
- Only one Transformer configuration was tested.
- The model was evaluated on one Python bug-fix dataset, so performance may not generalize to other datasets or languages.
- The positional encoding supports sequences up to `1,024` tokens.
- Future work could use controlled ablation studies to isolate the effect of sequence context, compare model depths, and investigate strategies for inputs longer than `1,024` tokens.

## Tech Stack

* Python
* pandas
* NumPy
* scikit-learn
* PyTorch
* Matplotlib
* Hugging Face Datasets
* Jupyter
* PyArrow

Planned deployment tools:

* FastAPI
* Docker

---

## Repository Structure

```text
notebooks/
├── v1/
│   ├── 01_preprocessing.ipynb
│   ├── 02_create_splits.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_error_analysis.ipynb
│   └── 05_dataset_explorations.ipynb
├── v2/
│   ├── 01_creating_validation.ipynb
│   ├── 02_tokenizer.ipynb
│   ├── 03_vocabulary.ipynb
│   ├── 04_model.ipynb
│   └── 05_evaluation.ipynb
└── v3/
    ├── 01_transformer_model.ipynb
    ├── 02_evaluation_v2_v3.ipynb
    └── 03_final_evaluation.ipynb

src/
├── preprocessing.py
├── v2/
│   ├── comment_only_diff.py
│   ├── data.py
│   ├── evaluate.py
│   ├── ids_and_tokens.py
│   ├── tokenize_diff.py
│   └── train.py
└── v3/
    └── model.py

models/
├── v2_best_model.pt
└── v3_best_model.pt
```

---

## Reproducing the Full Experiment

Install the required dependencies from the repository root:

```bash
pip install -r requirements.txt
```

The raw and processed datasets are not committed to the repository. Run the required notebooks in the following order:

1. `notebooks/v1/01_preprocessing.ipynb`
2. `notebooks/v1/02_create_splits.ipynb`
3. `notebooks/v1/03_baseline_models.ipynb`
4. `notebooks/v2/01_creating_validation.ipynb`
5. `notebooks/v2/02_tokenizer.ipynb`
6. `notebooks/v2/03_vocabulary.ipynb`
7. `notebooks/v2/04_model.ipynb`
8. `notebooks/v2/05_evaluation.ipynb`
9. `notebooks/v3/01_transformer_model.ipynb`
10. `notebooks/v3/02_evaluation_v2_v3.ipynb`
11. `notebooks/v3/03_final_evaluation.ipynb`

The V1 error-analysis and dataset-exploration notebooks are not required to train the final models, but they contain additional analysis:

* `notebooks/v1/04_error_analysis.ipynb`
* `notebooks/v1/05_dataset_explorations.ipynb`

The V2 evaluation notebook also creates the fairly retrained V1 vectorizer and Logistic Regression artifacts used by the final evaluation. These generated `.joblib` files are excluded from version control.

The test split should remain untouched through the V3 validation comparison. The final evaluation notebook is evaluation-only and should not be used to make further model or hyperparameter changes.

The preprocessing notebook may require Hugging Face authentication to access RunBugRun.
