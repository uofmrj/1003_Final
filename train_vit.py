import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
import logging
import torch.nn as nn
from torchvision import models
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
import timm

logging.basicConfig(level=logging.INFO, format=' %(message)s')

class TripletDataset(Dataset):
    def __init__(self, dataset_path, num_classes, transform=None):
        self.dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
        self.class_to_indices = self._map_indices_by_class(num_classes)

    def _map_indices_by_class(self, num_classes):
        # Filter indices for images belonging to the first 'num_classes' classes
        indices = [i for i, (_, label) in enumerate(self.dataset.samples) if label < num_classes]
        # Organize indices by class
        class_to_indices = {}
        for idx in indices:
            _, label = self.dataset.samples[idx]
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)
        return class_to_indices

    def __getitem__(self, index):
        # Choose a random class for anchor and positive
        class_idx = random.choice(list(self.class_to_indices.keys()))
        # Choose two different images (anchor, positive) from the same class
        anchor_idx, positive_idx = random.sample(self.class_to_indices[class_idx], 2)
        # Choose a negative class different from anchor's class
        negative_class_idx = random.choice(list(self.class_to_indices.keys()))
        while negative_class_idx == class_idx:
            negative_class_idx = random.choice(list(self.class_to_indices.keys()))
        # Choose a random image from the negative class
        negative_idx = random.choice(self.class_to_indices[negative_class_idx])

        anchor_image, _ = self.dataset[anchor_idx]
        positive_image, _ = self.dataset[positive_idx]
        negative_image, _ = self.dataset[negative_idx]

        return anchor_image, positive_image, negative_image

    def __len__(self):
        return min(len(indices) for indices in self.class_to_indices.values()) * len(self.class_to_indices)

def get_dataloader(dataset_path, num_classes, batch_size, transform):
    dataset = TripletDataset(dataset_path, num_classes, transform)
    logging.info("DataLoader setup complete with batch size: %d", batch_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),          # Convert images to tensor
])





class FSCEmbNet(nn.Module):
    """Network for extracting features from images using a DINO ViT Small backbone.
    This network uses a pretrained DINO ViT Small backbone and adds a fully connected (FC) layer to map images to a specified dimension feature vector.

    Args:
        embedding_dim (int): Dimension of the feature vector to which images are mapped.
    """

    def __init__(self, embedding_dim=128):
        super(FSCEmbNet, self).__init__()
        # Load a pretrained DINO ViT Small model
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True)
        # Remove the classifier head
        self.backbone.head = nn.Identity()
        self.fc = nn.Linear(in_features=768, out_features=embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)

        # Normalize the output vector
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x
    

def train_emb_model(epochs, batch_size, learning_rate, data_loader, embedding_dim=256, margin=0.3):
    emb_model = FSCEmbNet(embedding_dim=embedding_dim)
    emb_model.train()
    optimizer = torch.optim.Adam(emb_model.parameters(), lr=learning_rate)

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_model.to(device)

    for epoch in range(epochs):
        for batch_id, (anchors, positives, negatives) in enumerate(data_loader):
            anchors = anchors.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)

            optimizer.zero_grad()
            anchors_embedding = emb_model(anchors)
            positives_embedding = emb_model(positives)
            negatives_embedding = emb_model(negatives)

            # Calculate pairwise distance between embeddings
            positive_similarity = F.cosine_similarity(anchors_embedding, positives_embedding)
            negative_similarity = F.cosine_similarity(anchors_embedding, negatives_embedding)
            # Calculate triplet loss
            losses = torch.relu(margin - (positive_similarity - negative_similarity))
            loss = losses.mean()

            if (batch_id + 1) % batch_size == 0:
                print(f"Epoch: {epoch + 1}, Batch: {batch_id + 1}, Loss: {loss.item()}")

            loss.backward()
            optimizer.step()

    torch.save(emb_model.state_dict(), 'models/emb_model.pth')

def create_emb_model_data_reader(chosen_classes, batch_size):
    # Placeholder for actual data loading logic.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = TripletDataset(r'/home/jiarj2402/1003/CUB_200_2011/images', chosen_classes, transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

def main():
    # Training parameters
    emb_epochs = 10
    emb_batch_size = 32
    emb_learning_rate = 0.0005
    chosen_classes = 100
    # Create training data reader
    emb_data_loader = create_emb_model_data_reader(chosen_classes=chosen_classes, batch_size=emb_batch_size)
    # Start training the model
    train_emb_model(epochs=emb_epochs, batch_size=emb_batch_size, learning_rate=emb_learning_rate, data_loader=emb_data_loader)
    

if __name__ == '__main__':
    main()