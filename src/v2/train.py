import torch.nn as nn
import torch

class BugFixClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dimension, num_classes, padding_index):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dimension, padding_idx=padding_index)
        self.classifier = nn.Linear(embedding_dimension, num_classes) #Layer creation

    def forward(self, tokens, padding_mask):
        embedded_tokens = self.embeddings(tokens) # starts as: (batchsize, maxlen) -> (batchsize, maxlen, embeddingdim)
        real_token_mask = (~padding_mask).int() #starts as: (batchsize, maxlen) -> (batchsize, maxlen) results in 1s/0s each diff has a vector of 1s/0s rather than the True and False
        unsqueezed = real_token_mask.unsqueeze(dim=2) #starts as: (batchsize, maxlen) -> (batchsize, maxlen, 1) for broadcasting later
        masked = embedded_tokens * unsqueezed #results: (batchsize, maxlen, embeddingdim) retains values for embeddings that are real tokens. pads are reduced to 0

        token_sums = masked.sum(dim=1) #sum all of the embeddings
        token_counts = unsqueezed.sum(dim=1) #sum all of the real tokens (True = 1, False = 0)

        if (token_counts == 0).any():
            raise ValueError('Cannot mean-pool a sequence containing no real tokens.')

        mean_pooling = token_sums / token_counts #results: (batchsize, 128) results in a tensor using broadcasting
        logits = self.classifier(mean_pooling) #results: (batchsize, scores) pass data through the layer
        return logits

def train_model(model, train_loader, validation_loader, criterion, optimizer, epochs, checkpoint_path='models/v2_best_model.pt', print_stats=True):
    #Pipeline Overview:
    #1 Predict 
    #2 Compute Loss (criterion)
    #3 Apply Backpropagation
    #4 Update Weights (Optimzer)

    history = {'train_loss': [], 'train_accuracy': [], 'validation_loss': [], 'validation_accuracy': []}
    best_validation_loss = float('inf') #used for saving the best model later
    for epoch in range(epochs):
        model.train() #put the model into train mode
        total_loss = 0
        total_correct = 0
        total_examples = 0

        for tokens, labels, padding_mask in train_loader:
            optimizer.zero_grad() #resets the gradient

            logits = model(tokens, padding_mask) #forward pass
            loss = criterion(logits, labels) #compute loss
            
            loss.backward() #apply backpropagation
            optimizer.step() #change weights

            predictions = logits.argmax(dim=1) #take the largest score
            total_correct += (predictions == labels).sum().item()
            total_examples += labels.size(0)
            total_loss += loss.item() * labels.size(0)

        train_loss = total_loss / total_examples
        train_accuracy = total_correct / total_examples

        model.eval() #put the model into evaluation mode

        validation_loss = 0
        validation_correct = 0
        validation_examples = 0

        with torch.no_grad(): #no gradient needed since no weights are changing
            for tokens, labels, padding_mask in validation_loader:
                logits = model(tokens, padding_mask) #forward pass
                loss = criterion(logits, labels)  #compute loss

                predictions = logits.argmax(dim=1)
                validation_loss += loss.item() * labels.size(0)
                validation_correct += (predictions == labels).sum().item()
                validation_examples += labels.size(0)

        validation_epoch_loss = validation_loss / validation_examples
        validation_accuracy = validation_correct / validation_examples
        if validation_epoch_loss < best_validation_loss:
            best_validation_loss = validation_epoch_loss
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'validation_loss': validation_epoch_loss}, checkpoint_path)

        if print_stats:
            print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f} Validation Loss: {validation_epoch_loss:.4f} Validation Accuracy: {validation_accuracy:.4f}')

        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['validation_loss'].append(validation_epoch_loss)
        history['validation_accuracy'].append(validation_accuracy)

    return history