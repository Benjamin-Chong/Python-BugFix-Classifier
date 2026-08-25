# Python Bug-Fix Classifier

An end-to-end machine learning project that compares how three code representations affect Python bug-fix classification. Given buggy and fixed code, the application generates a diff and predicts one of five bug-fix categories with each model:

- **V1:** TF-IDF with logistic regression
- **V2:** Learned token embeddings with masked mean pooling
- **V3:** A sequence-aware Transformer encoder with positional encoding and self-attention

The three frozen models are exposed through a FastAPI inference service and an interactive HTML, CSS, and JavaScript frontend.

**[Try the live application](https://python-bugfix-classifier-api.onrender.com/)** · **[Open the API documentation](https://python-bugfix-classifier-api.onrender.com/docs)**

> **Note:** The free Render service may take a short time to wake after a period of inactivity.

## Results at a Glance

All three models were evaluated on the same 7,129-example test split.

| Model | Test Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| V1: TF-IDF + logistic regression | 70.68% | 58.88% | 67.55% |
| V2: embeddings + mean pooling | 77.25% | 71.20% | 76.87% |
| V3: Transformer encoder | **87.52%** | **83.84%** | **87.25%** |

V3 achieved the strongest overall performance and improved every class-level F1 score over V2. The largest gains occurred in `expression` and `identifier`, the two categories expected to benefit most from token order and contextual relationships.

## Table of Contents

- [Project Motivation](#project-motivation)
- [Dataset and Labels](#dataset-and-labels)
- [Experimental Design](#experimental-design)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Vocabulary and Sequence Design](#vocabulary-and-sequence-design)
- [Model Architectures](#model-architectures)
- [Training Configuration](#training-configuration)
- [Evaluation Results](#evaluation-results)
- [Error Analysis](#error-analysis)
- [Web Application and Deployment](#web-application-and-deployment)
- [Limitations and Future Work](#limitations-and-future-work)
- [Tech Stack](#tech-stack)
- [Local Installation and API Usage](#local-installation-and-api-usage)
- [Reproducing the Full Experiment](#reproducing-the-full-experiment)
- [Repository Structure](#repository-structure)
- [Data Source and Citation](#data-source-and-citation)

## Project Motivation

This project investigates how different code representations affect bug-fix classification.

V1 established an interpretable baseline by comparing logistic regression and LinearSVC using TF-IDF features. Logistic regression performed slightly better and was selected as the final V1 classifier. Error analysis showed that TF-IDF could recognize useful token associations but struggled when classification depended on token meaning, order, or surrounding context.

These findings motivated a controlled progression:

1. Replace fixed TF-IDF features with task-specific learned embeddings in V2.
2. Preserve token order and model token-to-token relationships with positional encoding and self-attention in V3.
3. Compare all three frozen models through one deployed inference application.

## Dataset and Labels

The project uses Python examples from [RunBugRun](https://arxiv.org/abs/2304.01102). Each selected example provides buggy code, fixed code, and one or more hierarchical bug-type labels.

RunBugRun labels can contain multiple levels, such as `assignment.value.change`. For this project, each label was reduced to its top-level category, such as `assignment`, and only examples containing a single label were retained. Classes with fewer than 200 examples were removed rather than combined into an artificial miscellaneous category.

| Filtering Stage | Examples |
|---|---:|
| Original Python examples | 133,705 |
| After retaining single-label examples | 35,926 |
| After removing underrepresented classes | 35,641 |

The removed classes were `literal`, `function`, `variable_access`, `io`, and `try_catch`. The remaining examples belong to five mutually exclusive target classes:

| Label | Project Definition |
|---|---|
| `assignment` | Changes involving assignment statements or assigned values |
| `call` | Changes involving function or method calls |
| `control_flow` | Changes involving conditions, loops, or other control-flow statements |
| `expression` | Changes involving expressions or operators |
| `identifier` | Changes involving variable, function, or other identifier names |

These five labels are a project-specific simplification of RunBugRun's complete hierarchical, multi-label taxonomy.

## Experimental Design

The original cleaned data was divided into 28,512 training examples and 7,129 test examples during V1. When V2 development began, the original training portion was split again to introduce a validation set:

| Split | Examples | Purpose |
|---|---:|---|
| Training | 22,809 | Fit model parameters, TF-IDF features, and vocabulary |
| Validation | 5,703 | Select vocabulary settings, models, and checkpoints |
| Test | 7,129 | Compare the frozen V1, V2, and V3 models |

The development split used stratification and a fixed random seed of `42` to preserve class distributions and make the experiment reproducible.

Several controls were used to make the architecture comparison fair:

- The fair-comparison V1 model was retrained on the same 22,809 examples used by V2 and V3.
- The V1 TF-IDF vectorizer was fitted using only the training split.
- The V2/V3 vocabulary was constructed using only training examples.
- V2 and V3 used the same tokenizer, vocabulary, numericalized inputs, and data splits.
- V2 and V3 checkpoints were selected using validation loss rather than test performance.
- All three models were frozen before the final common test-set comparison.

The test split had already been evaluated during the original V1 phase, before the validation protocol was introduced. V2 and V3 development did not use the test set, but this earlier exposure means the final comparison is not based on a completely untouched test set.

## Preprocessing Pipeline

### 1. Generate the code diff

- Split the buggy and fixed code into lines.
- Generate the diff with `difflib.ndiff`.
- Retain only added and removed lines.
- Discard unchanged lines and `?` hint lines to reduce noise.

### 2. Create the V1 representation

- Pass the filtered diff text into TF-IDF.
- Convert each diff into a sparse vector of weighted token features.
- Train logistic regression and LinearSVC on the same fixed representation.

### 3. Tokenize for V2 and V3

- Replace diff prefixes with `<ADD>` and `<DELETE>` markers.
- Preserve indentation using tokens such as `<INDENT_0>`.
- Use Python's `tokenize` module to separate identifiers, operators, keywords, and literals.
- Ignore comments and newline-control tokens.

### 4. Numericalize tokens and labels

- Construct the vocabulary from training tokens only.
- Reserve ID `0` for `<PAD>` and ID `1` for `<UNK>`.
- Convert tokens and class labels into integer IDs.

### 5. Prepare PyTorch batches

- Dynamically pad sequences to the longest sequence in each batch.
- Create masks so padding does not contribute to mean pooling or self-attention.

V2 and V3 use the same numericalized input pipeline, allowing the experiment to compare their architectures rather than different preprocessing strategies.

## Vocabulary and Sequence Design

Vocabulary frequency was measured using document frequency, where each tokenized diff represents one document. The selected threshold was `min_freq=4`: a token had to appear in at least four training examples unless it was a required structural token such as `<ADD>`, `<DELETE>`, or an indentation marker.

| Vocabulary Statistic | Value |
|---|---:|
| Final vocabulary size, including `<PAD>` and `<UNK>` | 947 |
| Unique training token types removed by the threshold | 81.67% |
| Validation out-of-vocabulary rate | 2.04% |

The low validation out-of-vocabulary rate showed that most removed token types were rare while the retained vocabulary still covered nearly all validation tokens.

The project initially considered truncating long diffs to reduce computation. Manual inspection showed that the label-defining change could appear at the beginning, middle, or end of a diff, so truncation could remove the actual fix. V2 therefore retained complete token sequences and used dynamic batch padding. V3's positional encoding supports a maximum of 1,024 tokens, which is also enforced by the deployed application.

## Model Architectures

| Version | Representation | Sequence-Aware | Classifier |
|---|---|:---:|---|
| V1 | TF-IDF | No | Logistic regression |
| V2 | Learned embeddings with masked mean pooling | No | Linear layer |
| V3 | Positional encoding and Transformer encoder | Yes | Linear layer |

### V1: TF-IDF + Logistic Regression

```text
Filtered diff
→ TF-IDF representation
→ Logistic Regression
→ five class probabilities
```

TF-IDF creates a sparse representation of each diff. Logistic regression and LinearSVC were compared, with logistic regression selected as the final V1 model. V1 is fast and directly interpretable through feature weights, but it cannot learn token meaning, order, or surrounding context.

### V2: Learned Embeddings + Masked Mean Pooling

```text
Token IDs
→ 128-dimensional embeddings
→ padding mask
→ masked mean pooling
→ Linear(128, 5)
→ five class logits
```

V2 learns a 128-dimensional embedding for every vocabulary token. After padding is masked, the remaining embeddings are averaged into one fixed-size vector and passed to a linear classifier. Unlike TF-IDF, the representation is learned from the classification task. However, mean pooling removes token order, so sequences containing the same tokens in different orders produce the same pooled representation.

### V3: Transformer Encoder

```text
Token IDs
→ 128-dimensional embeddings
→ sinusoidal positional encoding
→ two Transformer encoder layers
→ masked mean pooling
→ Linear(128, 5)
→ five class logits
```

V3 uses the same tokens, vocabulary, and embedding dimension as V2, then adds positional encoding and two Transformer encoder layers. Self-attention allows each token representation to incorporate information from other tokens before the contextualized sequence is pooled and classified.

The final configuration uses four attention heads, a feed-forward dimension of `256`, dropout of `0.1`, padding-aware attention, and a maximum positional length of `1,024`. V3 models order and token-to-token relationships but is more computationally expensive and less directly interpretable than V1.

## Training Configuration

V2 and V3 were trained from random initialization in PyTorch rather than fine-tuning a pretrained code model.

| Setting | V2 | V3 |
|---|---:|---:|
| Random seed | 42 | 42 |
| Batch size | 32 | 32 |
| Embedding dimension | 128 | 128 |
| Optimizer | Adam | Adam |
| Learning rate | 0.001 | 0.001 |
| Maximum epochs | 15 | 15 |
| Loss function | Cross-entropy | Cross-entropy |
| Checkpoint criterion | Lowest validation loss | Lowest validation loss |
| Selected epoch | 13 | 10 |
| Best validation loss | 0.5284 | 0.3144 |

The training loop recorded training and validation loss and accuracy after every epoch. The checkpoint with the lowest validation loss was saved for the final comparison.

### Validation Results

| Model | Validation Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| V1: TF-IDF + logistic regression | 70.58% | 58.91% | 67.69% |
| V2: embeddings + mean pooling | 77.80% | 72.49% | 77.48% |
| V3: Transformer encoder | **87.73%** | **83.56%** | **87.48%** |

The validation progression supported the original hypothesis without requiring test-set-driven model changes. V2 improved most over V1 on the weak `expression` and `identifier` categories, while V3 produced another substantial improvement after adding sequence-aware contextualization.

## Evaluation Results

The final comparison evaluated all three frozen models on the same 7,129-example test split.

| Model | Test Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| V1: TF-IDF + logistic regression | 70.68% | 58.88% | 67.55% |
| V2: embeddings + mean pooling | 77.25% | 71.20% | 76.87% |
| V3: Transformer encoder | **87.52%** | **83.84%** | **87.25%** |

V2 improved test accuracy by approximately 6.6 percentage points over V1, showing the benefit of learning task-specific token representations instead of relying only on fixed TF-IDF features.

V3 improved test accuracy by approximately 10.3 additional points over V2. Its macro F1 reached 83.84%, showing that its gains were not driven only by the majority class. Validation and test performance were also close, indicating that V3 remained stable on examples excluded from training.

These results support the hypothesis that sequence-aware representations are more effective than sparse token-frequency features or mean-pooled embeddings for this classification task. Because V3 also introduced additional capacity, the experiment does not prove that sequence awareness alone caused the improvement.

## Error Analysis

The class-level results show how each representation addressed weaknesses in the previous model:

| Class | V1 F1 | V2 F1 | V3 F1 |
|---|---:|---:|---:|
| `assignment` | 0.6363 | 0.7348 | **0.8334** |
| `call` | 0.8133 | 0.8401 | **0.9128** |
| `control_flow` | 0.8673 | 0.8920 | **0.9439** |
| `expression` | 0.2902 | 0.5703 | **0.7722** |
| `identifier` | 0.3367 | 0.5225 | **0.7297** |

### V1 Findings

V1 performed well on `control_flow`, reaching 85.73% recall. TF-IDF could associate the category with distinctive tokens such as `if`, `elif`, `while`, `break`, and `or`.

Its largest weaknesses were `expression` and `identifier`, with recalls of 20.70% and 23.15%. Of the 1,483 `expression` examples, V1 incorrectly classified 743 as `call` and 347 as `assignment`. These errors showed that TF-IDF could learn useful token associations but struggled when the correct label depended on meaning, order, or surrounding context.

The full V1 investigation is available in [`04_error_analysis.ipynb`](notebooks/v1/04_error_analysis.ipynb).

### V2 Findings

Learned embeddings improved every class. `expression` F1 increased from 0.2902 to 0.5703, while `identifier` F1 increased from 0.3367 to 0.5225.

However, these remained the two weakest categories. This was consistent with V2's architectural limitation: masked mean pooling combines the token embeddings into an average vector and cannot distinguish sequences containing the same tokens in different orders.

### V3 Findings

V3 improved all five class-level F1 scores. `expression` reached 0.7722 and `identifier` reached 0.7297. These gains are consistent with the hypothesis that positional encoding and self-attention help when classification depends on token order and relationships between tokens.

The main remaining weakness was `identifier` recall at 62.50%. It was also the smallest test category, with 216 examples, which may have limited the model's ability to learn its patterns.

Overall, the progression shows that learned representations improved on sparse token-frequency features and that the sequence-aware model produced the strongest results for the context-dependent categories.

## Web Application and Deployment

```text
Buggy and fixed code
→ FastAPI generates and validates the diff
→ model-specific preprocessing
→ V1, V2, and V3 inference
→ predictions and formatted diff returned to the frontend
```

The application is deployed as one Render web service. The frontend is built with HTML, CSS, and JavaScript, while FastAPI serves both the static pages and the inference API.

The `POST /predict` endpoint accepts `buggy_code` and `fixed_code`. The backend generates the diff, applies the required preprocessing, and runs all three frozen models. Its response contains the generated diff and one predicted label from each model.

The prediction page displays removed lines in red and added lines in green alongside the V1, V2, and V3 results. Inputs that produce no meaningful tokenized diff or exceed the 1,024-token limit are rejected with a `422 Unprocessable Entity` response.

## Limitations and Future Work

- The project reduces RunBugRun's hierarchical, multi-label taxonomy to five mutually exclusive top-level classes. The reported metrics therefore do not represent performance on the complete RunBugRun task.
- V3 introduced both sequence awareness and additional model capacity, so their individual effects were not isolated.
- `identifier` remained underrepresented, with only 216 examples in the test split.
- Only one Transformer configuration was evaluated, limiting conclusions about model depth, attention heads, embedding size, and other hyperparameters.
- The models were evaluated on a single Python bug-fix dataset, so performance may not generalize to other datasets, languages, or real-world bug distributions.
- The positional encoding supports sequences of up to 1,024 tokens, preventing the application from processing longer inputs.
- The test split was evaluated during V1 development before the validation protocol was introduced. V2 and V3 selection used only validation results, and all models were frozen before the final comparison, but the test set was not completely untouched throughout the project.
- The free Render service can introduce a cold start after inactivity.

Future work could use controlled ablation studies to separate sequence awareness from model capacity, evaluate additional Transformer configurations, investigate class-imbalance strategies, and test approaches for processing inputs longer than 1,024 tokens.

## Tech Stack

| Area | Technologies |
|---|---|
| Machine learning | PyTorch, scikit-learn |
| Data processing | pandas, NumPy, Hugging Face Datasets, PyArrow |
| Evaluation and analysis | scikit-learn metrics, Matplotlib, seaborn, Jupyter |
| Backend | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Render |

## Local Installation and API Usage

### Requirements

- Python 3.12
- Git

### Run the Application Locally

Clone the repository:

```bash
git clone https://github.com/Benjamin-Chong/Python-BugFix-Classifier.git
cd Python-BugFix-Classifier
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or activate it on macOS and Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the FastAPI application:

```bash
python -m uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000` for the web application or `http://127.0.0.1:8000/docs` for the interactive API documentation.

The frozen model and preprocessing artifacts required for inference are included in the repository. The original RunBugRun dataset is not required to run the application.

### API Usage

The `POST /predict` endpoint accepts buggy and fixed Python code:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "buggy_code": "total += prices",
    "fixed_code": "total += price"
  }'
```

The response contains the generated diff and the predicted label from V1, V2, and V3.

## Reproducing the Full Experiment

Running the web application and reproducing the complete training workflow are separate processes. The repository includes the compact mappings and frozen artifacts needed for inference, but it does not include the raw RunBugRun data or all intermediate processed datasets.

Install the dependencies from the repository root, obtain access to RunBugRun through Hugging Face if required, and execute the notebooks in this order:

<details>
<summary>View notebook execution order</summary>

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

</details>

The V1 error-analysis and dataset-exploration notebooks are optional for training but document additional investigation:

- `notebooks/v1/04_error_analysis.ipynb`
- `notebooks/v1/05_dataset_explorations.ipynb`

The V2 evaluation notebook creates the fairly retrained V1 TF-IDF vectorizer and logistic regression artifacts used in the final comparison. The final evaluation notebook compares the frozen V1, V2, and V3 models on the common test split; its results should not be used to make further model or hyperparameter changes.

## Repository Structure

```text
api/                 FastAPI routes and inference service
data/processed/      Compact vocabulary and label mappings used for inference
frontend/            Home, prediction, and about pages
models/              Frozen V1, V2, and V3 inference artifacts
notebooks/
├── v1/              TF-IDF baseline, exploration, and error analysis
├── v2/              Tokenizer, vocabulary, embedding model, and evaluation
└── v3/              Transformer model and final comparisons
src/
├── preprocessing.py Shared diff generation
├── v2/              Tokenization, datasets, training, and evaluation utilities
└── v3/              Transformer architecture
requirements.txt     Python dependencies
```

## Data Source and Citation

This project uses [RunBugRun](https://github.com/giganticode/run_bug_run), accessed through Hugging Face. If you use the dataset or build on this work, cite the original paper:

> Julian Aron Prenner and Romain Robbes. "RunBugRun — An Executable Dataset for Automated Program Repair." arXiv:2304.01102, 2023. [Paper](https://arxiv.org/abs/2304.01102)

```bibtex
@misc{prenner2023runbugrun,
  title        = {RunBugRun -- An Executable Dataset for Automated Program Repair},
  author       = {Julian Aron Prenner and Romain Robbes},
  year         = {2023},
  eprint       = {2304.01102},
  archivePrefix = {arXiv},
  primaryClass = {cs.SE},
  url          = {https://arxiv.org/abs/2304.01102}
}
```
