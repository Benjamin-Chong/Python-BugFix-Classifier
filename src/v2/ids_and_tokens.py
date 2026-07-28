def create_token_to_id(all_tokens): #expecting a dictionary
    mapping = {'<PAD>':0, '<UNK>':1}
    tokens = set(all_tokens)

    for token_id, token in enumerate(sorted(tokens), start=2):
        mapping[token] = token_id

    return mapping

def create_id_to_token(token_to_id_mapping):
    id_to_token_mapping = {value: key for key, value in token_to_id_mapping.items()}
    return id_to_token_mapping