
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
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
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, channels_img * image_size * image_size),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), channels_img, image_size, image_size)
        return x

# Discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(channels_img * image_size * image_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

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
additional_epochs = 30  # Set the number of additional epochs

# Continue the training loop
for epoch in range(num_epochs, num_epochs + additional_epochs):
    for batch_idx, (real, _) in enumerate(train_loader):
        real = real.to(device)
        batch_size = real.size(0)

        # Train Discriminator
        noise = torch.randn(batch_size, latent_dim).to(device)
        fake = generator(noise)
        labels_real = torch.ones(batch_size, 1).to(device)
        labels_fake = torch.zeros(batch_size, 1).to(device)

        outputs_real = discriminator(real)
        loss_real = criterion(outputs_real, labels_real)
        
        outputs_fake = discriminator(fake.detach())
        loss_fake = criterion(outputs_fake, labels_fake)
        
        loss_D = (loss_real + loss_fake) / 2
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        outputs_fake = discriminator(fake)
        loss_G = criterion(outputs_fake, labels_real)
        
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs + additional_epochs}] Batch {batch_idx}/{len(train_loader)} \
                  Loss D: {loss_D:.4f}, loss G: {loss_G:.4f}")

    # Save some sample images
    with torch.no_grad():
        fake_images = generator(noise).detach().cpu()
    save_image(fake_images, f"/project/jacobcha/nk643/gans/output/testing/epoch_{epoch}.png", nrow=12)

# Optionally, save the model again after additional training
torch.save(generator.state_dict(), '/project/jacobcha/nk643/gans/checkpoints/generator.pth')
torch.save(discriminator.state_dict(), '/project/jacobcha/nk643/gans/checkpoints/discriminator.pth')
