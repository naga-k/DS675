import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
latent_dim = 128
batch_size = 128
learning_rate = 0.0002
num_epochs = 20
image_size = 28
channels_img = 1

# TensorBoard Writer
writer = SummaryWriter('runs/fashion_gan')

# Image preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Load the Fashion MNIST dataset
train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Generator network
class Generator(nn.Module):
    # [Generator class definition...]
    pass

# Discriminator network
class Discriminator(nn.Module):
    # [Discriminator class definition...]
    pass

# Initialize networks
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Loss function
criterion = nn.BCELoss()

# Load the saved model weights
generator.load_state_dict(torch.load('/project/jacobcha/nk643/gans/checkpoints/generator.pth'))
discriminator.load_state_dict(torch.load('/project/jacobcha/nk643/gans/checkpoints/discriminator.pth'))

# Move models to the correct device
generator.to(device)
discriminator.to(device)

# Additional training epochs
additional_epochs = 30

# Continue the training loop
for epoch in range(num_epochs, num_epochs + additional_epochs):
    for batch_idx, (real, _) in enumerate(train_loader):
        # [Training loop code...]

        # Log the losses to TensorBoard
        writer.add_scalar('Loss/Discriminator', loss_D.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('Loss/Generator', loss_G.item(), epoch * len(train_loader) + batch_idx)

    # Save some sample images
    # [Image saving code...]

# Optionally, save the model again after additional training
torch.save(generator.state_dict(), '/project/jacobcha/nk643/gans/checkpoints/generator.pth')
torch.save(discriminator.state_dict(), '/project/jacobcha/nk643/gans/checkpoints/discriminator.pth')

# Close the writer
writer.close()


#tensorboard --logdir=runs
