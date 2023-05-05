import torch

def train(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch using the provided dataloader.

    Args:
    model: The PyTorch model being trained.
    dataloader: The DataLoader for the training dataset.
    criterion: The loss function.
    optimizer: The optimization algorithm.
    device: The device to run the training on (CPU or GPU).

    Returns:
    float: The average training loss for this epoch.
    float: The training accuracy for this epoch.
    """

    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate the number of correct predictions
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

    avg_loss = running_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions

    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    """
    Validate the model on the provided validation dataset.

    Args:
    model: The PyTorch model being validated.
    dataloader: The DataLoader for the validation dataset.
    criterion: The loss function.
    device: The device to run the validation on (CPU or GPU).

    Returns:
    tuple: (average validation loss, validation accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    return running_loss / len(dataloader), correct_predictions / total_predictions


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on the provided test dataset.

    Args:
    model: The PyTorch model being evaluated.
    dataloader: The DataLoader for the test dataset.
    criterion: The loss function.
    device: The device to run the evaluation on (CPU or GPU).

    Returns:
    float: The test loss.
    float: The test accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions

    return avg_loss, accuracy