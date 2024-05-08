import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import torch.nn as nn
from torchvision import models
from torchvision.utils import make_grid
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from torchvision.io import read_image
from collections import defaultdict
import torch.optim as optim
from collections import defaultdict
from sklearn.metrics import recall_score, f1_score
import timm
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

        anchor_image, anchor_label = self.dataset[anchor_idx]
        positive_image, _ = self.dataset[positive_idx]  # label is the same as the anchor
        negative_image, negative_label = self.dataset[negative_idx]

        return (anchor_image, positive_image, negative_image), (anchor_label, anchor_label, negative_label)

    def __len__(self):
        return min(len(indices) for indices in self.class_to_indices.values()) * len(self.class_to_indices)



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

# # Define the mean and std used for normalization
# mean = torch.tensor([0.485, 0.456, 0.406])
# std = torch.tensor([0.229, 0.224, 0.225])

# # Create a transform to unnormalize the image
# unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
# def show_image(image_tensor):
#     # Apply the unnormalization transformation
#     image_tensor = unnormalize(image_tensor)
#     # Clamp the image data to ensure it's within [0, 1] for display purposes
#     image_tensor = image_tensor.clamp(0, 1)
#     # Convert tensor for display: Remove batch dimension and permute dimensions
#     image_tensor = image_tensor.squeeze().permute(1, 2, 0)
#     plt.imshow(image_tensor)
#     plt.axis('off')  # Hide the axis
#     plt.show()

# def load_model(model_path, embedding_dim=128):
#     model = FSCEmbNet(embedding_dim=embedding_dim)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#     return model
# from torchvision.io import read_image
# def get_distance(embedding1, embedding2):
#     return torch.norm(embedding1 - embedding2, p=2, dim=1).item()
# def predict(model, image_tensor):
#     with torch.no_grad():
#         embedding = model(image_tensor)
#     return embedding

# def prepare_image(image_path, transform):
#     image_tensor = read_image(image_path)
#     image_tensor = transform(image_tensor)
#     return image_tensor.unsqueeze(0)  # Add batch dimension
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ConvertImageDtype(torch.float32),  # Ensure the tensor is in float format
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
# ])
# def run_comparison_demo(image_path_a, image_path_b, image_path_c, model_path):
#     # Load the model
#     model = load_model(model_path, embedding_dim=256)
    
#     # Prepare the images
#     image_tensor_a = prepare_image(image_path_a, transform)
#     image_tensor_b = prepare_image(image_path_b, transform)
#     image_tensor_c = prepare_image(image_path_c, transform)
    
#     # Compute embeddings
#     embedding_a = predict(model, image_tensor_a)
#     embedding_b = predict(model, image_tensor_b)
#     embedding_c = predict(model, image_tensor_c)
    
#     # Calculate distances
#     distance_ab = get_distance(embedding_a, embedding_b)
#     distance_ac = get_distance(embedding_a, embedding_c)

#     # Determine which image is closer to A
#     if distance_ab < distance_ac:
#         closer_image = "Image B is closer to Image A"
#     else:
#         closer_image = "Image C is closer to Image A"
    
#     print(f"Distance A-B: {distance_ab}, Distance A-C: {distance_ac}")
#     print(closer_image)

#     # Optionally display images
#     show_image(image_tensor_a.squeeze())
#     show_image(image_tensor_b.squeeze())
#     show_image(image_tensor_c.squeeze())

# # Example usage
# image_path_a = '/home/jiarj2402/1003/CUB_200_2011/images/041.Scissor_tailed_Flycatcher/Scissor_Tailed_Flycatcher_0019_41936.jpg'
# image_path_b = '/home/jiarj2402/1003/CUB_200_2011/images/041.Scissor_tailed_Flycatcher/Scissor_Tailed_Flycatcher_0007_41917.jpg'
# image_path_c = '/home/jiarj2402/1003/CUB_200_2011/images/026.Bronzed_Cowbird/Bronzed_Cowbird_0005_24173.jpg'
# model_path = '/home/jiarj2402/1003/models/emb_model.pth'
# run_comparison_demo(image_path_a, image_path_b, image_path_c, model_path)



def create_support_set_and_test_dataset(dataset_path, chosen_classes=5, num_shot=2, num_test=5):
    """
    Adjusted to load and transform images into tensors.
    """
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=dataset_path, transform=trans)
    # Only keeping the last chosen_classes
    classes = [dataset.classes[-i-1] for i in range(chosen_classes)]
    class_to_idx = {cls: idx for idx, cls in enumerate(dataset.class_to_idx) if cls in classes}

    # Filter dataset to only include images from the selected classes
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    filtered_indices = [idx for idx, (_, class_idx) in enumerate(dataset.samples) if class_idx in class_to_idx.values()]

    # Organize by class
    class_idx_to_images = defaultdict(list)
    for idx in filtered_indices:
        path, class_idx = dataset.samples[idx]
        image = dataset.loader(path)  # Load the image
        image = trans(image)  # Apply transformations
        class_idx_to_images[class_idx].append(image)

    # Prepare support and test sets
    support_set = defaultdict(list)
    test_set = defaultdict(list)
    m = min(class_idx_to_images)
    for class_idx, images in class_idx_to_images.items():
        assert len(images) >= (num_shot + num_test), 'Not enough images for class.'
        random.shuffle(images)
        support_set[class_idx-m] = images[:num_shot]
        test_set[class_idx-m] = images[num_shot:num_shot + num_test]

    return support_set, test_set


class FSCPredictor(nn.Module):
    def __init__(self, embedding_net, support_set):
        super(FSCPredictor, self).__init__()
        self.embedding_net = embedding_net.to('cuda')  # Move model to GPU
        self.embedding_net.eval()

        matrix_m = []
        with torch.no_grad():
            for class_id, images in support_set.items():
                images_tensor = torch.stack(images).to('cuda')  # Move images to GPU
                class_embs = self.embedding_net(images_tensor)
                this_emb = class_embs.mean(dim=0)
                matrix_m.append(this_emb)

        matrix_m = torch.stack(matrix_m)
        self.softmax_classifier_w = nn.Parameter(matrix_m.clone().detach())
        self.softmax_classifier_b = nn.Parameter(torch.zeros(matrix_m.size(0)).to('cuda'))

    def forward(self, x):
        x = x.to('cuda')  # Move input tensor to GPU
        return self._forward_impl(x)

    def _forward_impl(self, x):
        x = self.embedding_net(x)
        normed_w = self.softmax_classifier_w / self.softmax_classifier_w.norm(dim=1, keepdim=True)
        normed_w = normed_w.t()
        x = torch.matmul(x, normed_w) + self.softmax_classifier_b
        return x

import torch.optim as optim

def train_predictor(embedding_model, support_set, epochs, learning_rate, use_entropy_regularization=True):
    """
    Train a few-shot classification predictor.

    :param embedding_model: Network used for extracting image features.
    :param support_set: Support set for few-shot classification.
    :param epochs: Number of training epochs.
    :param learning_rate: Learning rate for the classifier.
    :param use_entropy_regularization: Whether to use entropy regularization.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = FSCPredictor(embedding_net=embedding_model, support_set=support_set).to(device)
    predictor.train()

    # Only learn parameters W and b in the Softmax classifier
    optimizer = optim.Adam([predictor.softmax_classifier_w, predictor.softmax_classifier_b], lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # Prepare the entire support set as training data
        input_datas = []
        labels = []
        for class_id, imgs in support_set.items():
            input_datas.extend(imgs)
            labels.extend([class_id] * len(imgs))  
        input_datas = torch.stack(input_datas).to(device)  # Convert list of tensors to a single tensor and move to the correct device
        labels = torch.tensor(labels, dtype=torch.long).to(device)  # Ensure labels are on the same device as input data
        logits = predictor(input_datas)
        loss = criterion(logits, labels)

        if use_entropy_regularization:
            # Calculate entropy regularization as cross-entropy of softmaxed logits with themselves
            softmaxed_logits = torch.softmax(logits, dim=1)
            entropy_regularization = -(softmaxed_logits * torch.log(softmaxed_logits + 1e-6)).sum(dim=1).mean()
            loss += entropy_regularization

        if (epoch + 1) % 20 == 0:
            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save trained parameters
    torch.save(predictor.state_dict(), 'models/predictor.pth')



def load_model(model_path, embedding_dim=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = FSCEmbNet(embedding_dim=embedding_dim).to(device=device)
    model.load_state_dict(torch.load(model_path))
    return model

def test(predictor, test_set):
    """
    Test the few-shot classification model.

    :param predictor: Few-shot classification predictor.
    :param test_set: Test set.
    """
    predictor.eval()  # Set the model to evaluation mode
    device = next(predictor.parameters()).device  # Get the device of the model parameters

    total_test_samples_num = 0
    total_right_samples_num = 0
    all_labels = []
    all_preds = []
    inference_times = []

    print('============================================================')

    with torch.no_grad():  # Disable gradient computation for inference
        for class_id, this_class_data in test_set.items():
            start_time = time.time()  # Start time for inference

            this_class_samples_num = len(this_class_data)
            total_test_samples_num += this_class_samples_num

            # Convert data list to tensor and transfer it to the correct device
            this_class_data = torch.stack(this_class_data).to(device)

            # Get predictions
            predict_this_class = predictor(this_class_data)
            elapsed_time = time.time() - start_time
            inference_time = elapsed_time / this_class_samples_num
            inference_times.append(inference_time)

            predict_label = torch.argmax(predict_this_class, dim=1).cpu().numpy()  # Move tensor to CPU before converting to numpy
            
            # Store predictions and labels for later metrics calculation
            all_labels.extend([class_id] * this_class_samples_num)
            all_preds.extend(predict_label.tolist())

            # Calculate correct predictions
            this_right_samples_num = np.sum(predict_label == class_id)
            total_right_samples_num += this_right_samples_num
            accuracy = this_right_samples_num / this_class_samples_num
            print(f'Label:{200 - class_id}, Samples:{this_class_samples_num}, Correct:{this_right_samples_num}, Accuracy:{accuracy:.2f}, Avg. inference time/image:{inference_time:.4f}s')

    # Calculate overall metrics
    overall_accuracy = total_right_samples_num / total_test_samples_num
    recall = recall_score(all_labels, all_preds, average='macro')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    avg_inference_time = np.mean(inference_times)

    print(f'Total samples: {total_test_samples_num}, Total correct: {total_right_samples_num}')
    print(f'Overall Accuracy: {overall_accuracy:.2f}')
    print(f'Macro Recall: {recall:.2f}')
    print(f'Macro F1 Score: {macro_f1:.2f}')
    print(f'Average Inference Time per Image: {avg_inference_time:.4f}s')



chosen_classes = 10
support_set, test_set = create_support_set_and_test_dataset('/home/jiarj2402/1003/CUB_200_2011/images', chosen_classes=chosen_classes, num_shot=6, num_test=3)
predictor_epochs = 500
predictor_learning_rate = 0.0005
# Load the pre-trained feature extraction network
emb_model = load_model('/home/jiarj2402/1003/models/emb_model.pth', embedding_dim=256)

# Train the predictor
train_predictor(emb_model, support_set, epochs=predictor_epochs, learning_rate=predictor_learning_rate)

embedding_model = FSCEmbNet(embedding_dim=256)
embedding_model.eval()

# Initialize the FSCPredictor with the loaded embedding model and the support set
predictor = FSCPredictor(embedding_net=embedding_model, support_set=support_set)

# Load the trained state dictionary (ensure it is saved in the correct format for PyTorch)
predictor_state_dict = torch.load('models/predictor.pth')
predictor.load_state_dict(predictor_state_dict)

# Perform the test using the loaded predictor and the prepared test set
test(predictor, test_set)