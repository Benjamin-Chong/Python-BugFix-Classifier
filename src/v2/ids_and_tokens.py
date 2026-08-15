def create_token_to_id(all_tokens): #accepts an iterable of retained tokens
    mapping = {'<PAD>':0, '<UNK>':1}
    tokens = set(all_tokens)

    for token_id, token in enumerate(sorted(tokens), start=2):
        mapping[token] = token_id

    return mapping 

def create_id_to_token(token_to_id_mapping):
    id_to_token_mapping = {value: key for key, value in token_to_id_mapping.items()}
    return id_to_token_mapping

def tokens_to_ids(row_tokens, all_tokens_to_id): #uses the tokens_to_id mapping to return the number
    result = []
    unknown_id = all_tokens_to_id['<UNK>']
    for token in row_tokens:
        if token in all_tokens_to_id:
            number = all_tokens_to_id[token]
            result.append(number)
        else:
            result.append(unknown_id) #ID number for unknown

    return result