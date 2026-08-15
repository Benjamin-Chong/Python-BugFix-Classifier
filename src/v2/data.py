import torch
from torch.utils.data import Dataset

class BugFixDataset(Dataset): #adapter that makes our pandas data understandable to PyTorch
    def __init__(self, dataframe):
        super().__init__()
        self.dataframe = dataframe
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, index):
        tokens_in_ids = self.dataframe['tokens_ids'].iloc[index]
        label_id = self.dataframe['label_id'].iloc[index]
        return torch.tensor(tokens_in_ids, dtype=torch.long), torch.tensor(label_id, dtype=torch.long)


def collate_batch(batch): #batch has tensors for tokens, labels. used to create a rectangular shape so matrix operations can be applied
    token_tensors, label_tensors = zip(*batch) #after zipping both return the tensors and the labels
    batch_max_length = 0 #find the max length for proper matrix multiplication (shape)
    for tokens in token_tensors: #find the max length
        batch_max_length = max(batch_max_length, len(tokens))

    padded_token_tensors = []
    for tokens in token_tensors: #adds padding where needed and puts it into the padded token tensor list
        amount_of_padding = batch_max_length - len(tokens)
        pads = torch.zeros(amount_of_padding, dtype=torch.long, device=tokens.device) #adds 0s
        padded_sequence = torch.cat((tokens, pads)) #torch.cat returns a new object
        assert torch.equal(padded_sequence[:len(tokens)], tokens) #ensures that the padded sequence real tokens and the original tokens are still equal
        padded_token_tensors.append(padded_sequence)

    stacked_padded_tokens = torch.stack(padded_token_tensors) #list -> 1 tensor
    labels = torch.stack(label_tensors) #tensors individually -> 1 tensor

    padding_mask = stacked_padded_tokens == 0 #creates a mask holding True for pads and False for real tokens

    return stacked_padded_tokens, labels, padding_mask