import torch

def evaluate_model(model, criterion, loader):
    total_loss = 0
    correct = 0
    examples = 0
    all_predictions = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for tokens, labels, padding_mask in loader:
            logits = model(tokens, padding_mask)
            batch_loss = criterion(logits, labels)
            

            predictions = logits.argmax(dim=1)
            total_loss += batch_loss.item() * labels.size(0)
            correct += (predictions == labels).sum().item()
            examples += labels.size(0)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    accuracy = correct / examples
    average_loss = total_loss / examples
    return accuracy, average_loss, all_predictions, all_labels