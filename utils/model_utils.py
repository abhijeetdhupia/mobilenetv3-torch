import os
import torch

def save_checkpoint(model, optimizer, epoch, loss, accuracy, checkpoint_dir):
    """
    Save a model checkpoint.

    Args:
    model: The PyTorch model to save.
    optimizer: The optimizer used during training.
    epoch: The current epoch number.
    loss: The loss value for the current epoch.
    accuracy: The accuracy value for the current epoch.
    checkpoint_dir: The directory where to save the checkpoint.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "accuracy": accuracy,
    }

    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pt")
    torch.save(checkpoint, checkpoint_file)

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """
    Load a model checkpoint.

    Args:
    model: The PyTorch model to load the checkpoint into.
    optimizer: The optimizer used during training.
    checkpoint_path: The path to the checkpoint file.
    device: The device to load the checkpoint on (CPU or GPU).

    Returns:
    tuple: (epoch, loss, accuracy) for the loaded checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    accuracy = checkpoint["accuracy"]

    return epoch, loss, accuracy
