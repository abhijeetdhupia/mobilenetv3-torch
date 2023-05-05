import torchvision
import numpy as np
import matplotlib.pyplot as plt

def show_images_grid(images, title):
    """
    Show a grid of images using matplotlib.

    Args:
    images: A batch of images as a PyTorch tensor.
    title: The title for the grid of images.
    """
    img_grid = torchvision.utils.make_grid(images)
    np_img = img_grid.numpy().transpose((1, 2, 0))
    plt.imshow(np_img)
    plt.title(title)
    plt.show()

def create_summary_writer(log_dir):
    """
    Create a dummy SummaryWriter that simply prints the log directory.

    Args:
    log_dir: The directory where to store the TensorBoard logs.

    Returns:
    A SummaryWriter object.
    """
    print("Creating SummaryWriter with log directory:", log_dir)
    class SummaryWriter:
        def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
            print("Adding image to SummaryWriter with tag:", tag, "and global step:", global_step)
        def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
            print("Adding scalar to SummaryWriter with tag:", tag, "and global step:", global_step)

    return SummaryWriter()

def log_image_grid(summary_writer, images, tag, global_step):
    """
    Print the tag, images and global step.

    Args:
    summary_writer: The TensorBoard SummaryWriter.
    images: A batch of images as a PyTorch tensor.
    tag: The tag for the images in TensorBoard.
    global_step: The global step (usually the epoch number).
    """
    print("Logging image grid with tag:", tag, "and global step:", global_step)
    print("Images:", images)

def log_scalar(summary_writer, tag, value, global_step):
    """
    Print the tag, value and global step.

    Args:
    summary_writer: The TensorBoard SummaryWriter.
    tag: The tag for the scalar value in TensorBoard.
    value: The scalar value to log.
    global_step: The global step (usually the epoch number).
    """
    print("Logging scalar with tag:", tag, "and global step:", global_step)
    print("Value:", value)
