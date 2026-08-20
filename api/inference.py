from pathlib import Path
from src.v2.train import BugFixClassifier
from src.v3.model import TransformerBugFixClassifier
from src.v2.tokenize_diff import tokenize_diff
from src.v2.ids_and_tokens import tokens_to_ids
import joblib
import torch
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOKEN_TO_ID_PATH = PROJECT_ROOT/'data'/'processed'/'token_to_id.json'
ID_TO_LABEL_PATH = PROJECT_ROOT/'data'/'processed'/'id_to_label.json'

V1_LOGISTIC_REGRESSION_NO_VALIDATION_PATH = PROJECT_ROOT/'models'/'logistic_regression_v1_no_validation.joblib'
V1_TFIDF_VECTORIZER_NO_VALIDATION_PATH = PROJECT_ROOT/'models'/'tfidf_vectorizer_v1_no_validation.joblib'

V2_BEST_MODEL_PATH = PROJECT_ROOT/'models'/'v2_best_model.pt'
V3_BEST_MODEL_PATH = PROJECT_ROOT/'models'/'v3_best_model.pt'


with open(TOKEN_TO_ID_PATH, 'r') as f:
    token_to_id = json.load(f)

with open(ID_TO_LABEL_PATH, 'r') as f:
    id_to_label = json.load(f)

v1_model = joblib.load(V1_LOGISTIC_REGRESSION_NO_VALIDATION_PATH)
v1_tfidf = joblib.load(V1_TFIDF_VECTORIZER_NO_VALIDATION_PATH)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
v2_model = BugFixClassifier(vocab_size=len(token_to_id), embedding_dimension=128, num_classes=len(id_to_label), padding_index=token_to_id['<PAD>']).to(device)
v2_checkpoint = torch.load(V2_BEST_MODEL_PATH, map_location=device, weights_only=True)
v2_model.load_state_dict(v2_checkpoint['model_state_dict'])
v2_model.to(device)
v2_model.eval()

v3_model = TransformerBugFixClassifier(vocab_size=len(token_to_id), embedding_dim=128, num_heads= 4, num_layers=2, feedforward_dim=256, num_classes=len(id_to_label), max_length=1024, padding_index=token_to_id['<PAD>'], dropout=0.1).to(device)
v3_checkpoint = torch.load(V3_BEST_MODEL_PATH, map_location=device, weights_only=True)
v3_model.load_state_dict(v3_checkpoint['model_state_dict'])
v3_model.to(device)
v3_model.eval()

def all_predictions(diff):
    predictions = {}
    #V1 Predictions
    transformed_diff = v1_tfidf.transform([diff])
    v1_class = v1_model.predict(transformed_diff)[0]
    predictions['v1_prediction'] = v1_class

    #V2/3 Predictions
    #1 tokenize the diff
    #2 convert into ids
    #3 predict
    #4 extract the scores and the prediction
    tokenized_diff = tokenize_diff(diff)
    ids_from_tokens = tokens_to_ids(tokenized_diff, token_to_id)
    token_tensor = torch.tensor(ids_from_tokens, dtype=torch.long, device=device).unsqueeze(0)
    padding_mask = token_tensor.eq(token_to_id['<PAD>'])
    if not ids_from_tokens:
        raise ValueError('The generated diff contains no tokens.')

    if len(ids_from_tokens) > 1024:
        raise ValueError('The generated diff exceeds the 1,024-token limit.')
    
    with torch.inference_mode():
        v2_raw_predictions = v2_model(token_tensor, padding_mask)
        v2_predicted_class = v2_raw_predictions.argmax(dim=1).item()

        v3_raw_predictions = v3_model(token_tensor, padding_mask)
        v3_predicted_class = v3_raw_predictions.argmax(dim=1).item()

        predictions['v2_prediction'] = id_to_label[str(v2_predicted_class)]
        predictions['v3_prediction'] = id_to_label[str(v3_predicted_class)]

    return predictions