from IPython.display import display, clear_output
import torchvision
import matplotlib.pyplot as plt

def show_tensor_images(image_tensor, num_images=4, size=(1, 28, 28)):
    clear_output(wait=True)  # Clear the output of the existing cell
    image_shifted = (image_tensor + 1) / 2
    image_unflat = image_shifted.detach().cpu()
    image_grid = torchvision.utils.make_grid(image_unflat[:num_images], nrow=3)
    plt.figure(figsize=(8, 8))
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis('off')
    plt.show()
