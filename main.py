import json
import torch
import torch.optim as optim
import torch.nn as nn

# Import functions from your utils and models modules
from utils import (get_data_transforms, get_data_loaders, train, validate, evaluate,
                   save_checkpoint, load_checkpoint, get_confusion_matrix,
                   get_classification_report, get_accuracy, get_precision_recall_f1)
from models import get_mobilenet_v3_large, get_mobilenet_v3_small


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

config = load_config('config.json')

# Load configuration
config = load_config('config.json')

# Set device
device = torch.device('cuda' if config['use_gpu'] and torch.cuda.is_available() else 'cpu')

# Prepare the data
data_transforms = get_data_transforms()


data_loader = get_data_loaders(config['dataset_dir'], data_transforms, batch_size=config['batch_size'])

train_loader = data_loader['train']
val_loader = data_loader['val']
test_loader = data_loader['test']

# Create the model
if config['model_type'] == 'mobilenet_v3_large':
    model = get_mobilenet_v3_large(pretrained=config['pretrained'])
elif config['model_type'] == 'mobilenet_v3_small':
    model = get_mobilenet_v3_small(pretrained=config['pretrained'])
else:
    raise ValueError("Invalid model type specified in the config file.")

# Update the last layer of the model for the desired number of classes
num_classes = config['num_classes']
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
model = model.to(device)

# Set up the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])

# Load checkpoint if needed
start_epoch = 0
if config.get('resume_checkpoint'):
    start_epoch, model, optimizer = load_checkpoint(config['resume_checkpoint'], model, optimizer, device)

# Main training loop
for epoch in range(start_epoch, config['num_epochs']):
    # Train the model
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)

    # Validate the model
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    # Print the training and validation loss and accuracy
    print(f"Epoch {epoch + 1}/{config['num_epochs']}:")
    print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    print(f"  Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    # Save a checkpoint
    # save_checkpoint(model, optimizer, epoch, val_loss, val_acc, config['checkpoint_dir'])

# Evaluate the model on the test dataset
test_loss, test_acc = evaluate(model, test_loader, criterion, device)

# Print the test loss and accuracy
print(f"Test Loss: {test_loss:.4f} Test Accuracy: {test_acc:.4f}")