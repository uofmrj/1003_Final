# Example command:
# python create_support_test.py -i ./CUB_200_2011/images

import numpy as np
import random
import torch
import pickle
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import argparse


def create_support_set_and_test_dataset(dataset_path, chosen_classes=10, num_shot=1, num_test=20):
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
    

def main(img_folder_path):
    support_set, test_set = create_support_set_and_test_dataset(img_folder_path, chosen_classes=10, num_shot=1, num_test=3)
    # Saving support
    with open('ft_set/support_set.pkl', 'wb') as file:
        pickle.dump(support_set, file)
    print("Defaultdict saved to support_set.pkl")

    # Saving test
    with open('ft_set/test_set.pkl', 'wb') as file:
        pickle.dump(test_set, file)
    print("Defaultdict saved to test_set.pkl")

# Example command:
# python create_support_test.py -i ./CUB_200_2011/images
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create Support & Test Set")
    parser.add_argument('-i', '--img_folder_path', type=str, required=True, help='Path to the image folder')

    args = parser.parse_args()
    main(args.img_folder_path)
