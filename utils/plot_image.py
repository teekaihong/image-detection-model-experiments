from matplotlib import pyplot as plt


def plot_image(image, title=None):
    """Plot an image with matplotlib.

    Args:
        image (numpy.ndarray): Image to plot.
        title (str): Title of the image.
    """
    plt.figure()
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.show()

def image_gallery(image_list:list[str], title=None, max_images:int=10, num_cols = 5):
    """Plot a list of images with matplotlib in `max_images` // `num_cols` + 1 rows and `num_cols` columns.

    Args:
        image_list (list): List of image paths.
        title (str): Title of the image.
        max_images (int): Maximum number of images to plot.
    """
    num_rows = (max_images // num_cols) + 1
    plt.figure(figsize=(num_cols * 3, num_rows * 3))
    num_images = min(len(image_list), max_images)
    for i in range(num_images):
        image = image_list[i]
        image = plt.imread(image)
        plt.subplot(num_images // num_cols + 1, num_cols, i + 1)
        plt.imshow(image)
        plt.axis('off')
        plt.tight_layout()
    if title is not None:
        plt.suptitle(title)
    plt.show()

def plot_reconstructions(comparisons_list:list[tuple], title=None, max_images:int=8):
    """Plot a list of image comparisons with matplotlib. Original images are on the left, reconstructions on the right.

    Args:
        comparisons_list (list): List of tuples of the form (original, reconstruction).
        title (str): Title of the image.
        max_images (int): Maximum number of images to plot.
    """
    comparisons_list = comparisons_list[:max_images]
    fig, axes = plt.subplots(len(comparisons_list), 2, figsize=(6, 3 * len(comparisons_list)), squeeze=False)
    for i, (original, reconstruction) in enumerate(comparisons_list):
        if i >= max_images:
            break
        axes[i, 0].imshow(original)
        axes[i, 1].imshow(reconstruction)
        if i == 0:
            axes[i, 0].set_title("Original")
            axes[i, 1].set_title("Reconstruction")
    if title is not None:
        fig.suptitle(title)
    # turn off axis
    for ax in axes.flatten():
        ax.axis('off')
    plt.tight_layout()
    plt.show()