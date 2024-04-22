import os
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError

# Function to rename .jpeg files to .jpg
def convert_jpeg_to_jpg(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.jpeg'):
                base = os.path.splitext(file)[0]
                old_file = os.path.join(root, file)
                new_file = os.path.join(root, base + '.jpg')
                # Check if the new file name already exists
                if not os.path.exists(new_file):
                    os.rename(old_file, new_file)
                    print(f'Renamed {old_file} to {new_file}')
                else:
                    print(f'Not renaming {old_file} because {new_file} already exists.')
                    
# Path to your image folder
image_folder_path = 'C:\\Users\\Lenovo\\Dropbox\\PC\\Desktop\\Proje'
convert_jpeg_to_jpg(image_folder_path)  # Convert .jpeg images to .jpg

# Custom dataset loader that skips unreadable files
class RobustImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform):
        super(RobustImageFolder, self).__init__(root, transform)
        # Filter out the files that are not images or are corrupted
        self.valid_files = []
        self.valid_targets = []
        for i, (file, target) in enumerate(self.samples):
            try:
                with open(file, 'rb') as f:
                    Image.open(f).convert('RGB')  # Try to open the file to check if it's valid
                self.valid_files.append(file)
                self.valid_targets.append(target)
            except (IOError, UnidentifiedImageError):
                print(f"Skipping file {file}, as it's not readable.")

        self.samples = [(self.valid_files[i], self.valid_targets[i]) for i in range(len(self.valid_files))]
        self.imgs = self.samples  # Compatibility with torchvision 0.2.2

# Define the transformation with data augmentation
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.RandomHorizontalFlip(),  # Randomly flip the images horizontally
    transforms.RandomRotation(10),  # Randomly rotate the images by up to 10 degrees
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly change brightness, contrast, saturation, and hue
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize tensors
])

# Load the dataset from folder using the robust loader
dataset = RobustImageFolder(root=image_folder_path, transform=transform)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=120, shuffle=True, drop_last=True)

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(-1, 128*128*3)  # Flatten the images to match the discriminator input
        return self.main(x)

# Hyperparameters
input_dim = 100  # Adjusted to a common starting point for GANs
output_dim = 128 * 128 * 3  # Adjusted to match the flattened image size
learning_rate_g = 0.0002  # Generator learning rate
learning_rate_d = 0.0001  # Discriminator learning rate
batch_size = 120  # Batch size
epochs = 400  # Number of epochs

# Model initialization
generator = Generator(input_dim, output_dim)
discriminator = Discriminator(128*128*3)  # Correct dimension for flattened images
criterion = nn.BCELoss()
optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate_g)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate_d)

# Training loop
for epoch in range(epochs):
    for i, data in enumerate(dataloader):
        if data is None:  # Skip the None values returned by RobustImageFolder
            continue
        images, _ = data
        current_batch_size = images.size(0)

        # Flatten images for processing in the discriminator
        images = images.view(current_batch_size, -1)

        # Generate real and fake labels
        real_labels = torch.ones(current_batch_size, 1)
        fake_labels = torch.zeros(current_batch_size, 1)

        # Forward pass for real images through discriminator
        optimizer_d.zero_grad()
        outputs_real = discriminator(images)
        loss_real = criterion(outputs_real, real_labels)

        # Generate fake images
        noise = torch.randn(current_batch_size, input_dim)
        fake_images = generator(noise)
        outputs_fake = discriminator(fake_images.detach())
        loss_fake = criterion(outputs_fake, fake_labels)

        # Backprop and optimize for discriminator
        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()
        outputs_fake = discriminator(fake_images)
        loss_g = criterion(outputs_fake, real_labels)
        loss_g.backward()
        optimizer_g.step()

        # Logging
        if (i+1) % 100 == 0 or (i+1) == len(dataloader):
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss D: {loss_d.item()}, Loss G: {loss_g.item()}')

# After training, visualize some generated images
with torch.no_grad():
    test_noise = torch.randn(batch_size, input_dim)
    generated_images = generator(test_noise).detach().cpu()
    generated_images = generated_images.view(generated_images.size(0), 3, 128, 128)
    generated_images = (generated_images + 1) / 2  # Undo normalization for visualization
    for i in range(5):  # Display 5 images
        plt.imshow(np.transpose(generated_images[i], (1, 2, 0)))
        plt.show()
