# exmaple commands:
# python res.py -sp ./support_set.pkl -tp ./test_set.pkl -m models/emb_model.pth -s cosine
# python res.py -sp ./support_set.pkl -tp ./test_set.pkl -m models/emb_model.pth -s l1 -e renyi
# python res.py -sp ./support_set.pkl -tp ./test_set.pkl -m models/emb_model.pth -s l2 -e shannon
import time
import numpy as np
import random
import logging
import torch
import pickle
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import models
from torchvision.utils import make_grid
from torch.optim import Adam
import torch.nn.functional as F
from torchvision.io import read_image
from collections import defaultdict
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import roc_curve, auc, recall_score, f1_score, precision_score
from sklearn.preprocessing import label_binarize
import argparse
from sklearn.metrics import recall_score, f1_score

class FSCEmbNet(nn.Module):
    """Network for extracting features from images
    This network uses a pretrained resnet18 backbone and adds a fully connected (FC) layer to map images to a specified dimension feature vector.

    Args:
        embedding_dim (int): Dimension of the feature vector to which images are mapped.
    """
    def __init__(self, embedding_dim=128):
        super(FSCEmbNet, self).__init__()
        # Load a pretrained ResNet-18 model
        rn18 = models.resnet18(weights='IMAGENET1K_V1')

        # Extract the backbone from the pretrained resnet18 model
        # Instead of modifying the original resnet18, we create a new Sequential module
        # that follows the same architecture minus the original fully connected layer
        self.backbone = nn.Sequential(
            rn18.conv1,
            rn18.bn1,
            rn18.relu,
            rn18.maxpool,
            rn18.layer1,
            rn18.layer2,
            rn18.layer3,
            rn18.layer4,
            rn18.avgpool,
            nn.Flatten()
        )

        # Define the new fully connected layer that maps to the embedding dimension
        self.fc = nn.Linear(in_features=512, out_features=embedding_dim)  # in_features depends on rn18's last layer output

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)

        # Normalize the output vector
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x


class FSCPredictor(nn.Module):
    def __init__(self, embedding_net, support_set, similarity):
        super(FSCPredictor, self).__init__()
        self.similarity = similarity
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_net = embedding_net.to(self.device)  # Move model to GPU
        self.embedding_net.eval()
        
        matrix_m = []
        with torch.no_grad():
            for class_id, images in support_set.items():
                images_tensor = torch.stack(images).to(self.device)  # Move images to GPU
                class_embs = self.embedding_net(images_tensor)
                this_emb = class_embs.mean(dim=0)
                matrix_m.append(this_emb)

        matrix_m = torch.stack(matrix_m)
        self.softmax_classifier_w = nn.Parameter(matrix_m.clone().detach())
        self.softmax_classifier_b = nn.Parameter(torch.zeros(matrix_m.size(0)).to(self.device))

    def forward(self, x):
        x = x.to(self.device)  # Move input tensor to GPU
        if self.similarity == 'cosine':
            return self._forward_impl_cosine(x)
        elif self.similarity == 'l2':
            return self._forward_impl_l2(x)
        elif self.similarity == 'l1':
            return self._forward_impl_l1(x)

    def _forward_impl_cosine(self, x):
        x = self.embedding_net(x)
        normed_w = self.softmax_classifier_w / self.softmax_classifier_w.norm(dim=1, keepdim=True)
        normed_w = normed_w.t()
        x = torch.matmul(x, normed_w) + self.softmax_classifier_b
        return x
    
    def _forward_impl_l2(self, x):
        x = self.embedding_net(x)
        normed_w = self.softmax_classifier_w / self.softmax_classifier_w.norm(dim=1, keepdim=True)
        distances = torch.sqrt(torch.sum((x.unsqueeze(1) - self.softmax_classifier_w.unsqueeze(0))**2, dim=-1))
        logits = -distances + self.softmax_classifier_b # negative distance as similarity
        return logits
    
    def _forward_impl_l1(self, x):
        x = self.embedding_net(x)
        normed_w = self.softmax_classifier_w / self.softmax_classifier_w.norm(dim=1, keepdim=True)
        distances = torch.sum(torch.abs(x.unsqueeze(1) - self.softmax_classifier_w.unsqueeze(0)), dim=-1)
        logits = -distances + self.softmax_classifier_b # negative distance as similarity
        return logits
    

def train_predictor(embedding_model, support_set, epochs, learning_rate, 
                    similarity, entropy_regularization=None, alpha=2, is_baseline=False):
    """
    Train a few-shot classification predictor.

    :param embedding_model: Network used for extracting image features.
    :param support_set: Support set for few-shot classification.
    :param epochs: Number of training epochs.
    :param learning_rate: Learning rate for the classifier.
    :param use_entropy_regularization: Whether to use entropy regularization.

    :return predictor: the predictor
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    predictor = FSCPredictor(embedding_net=embedding_model, 
                             support_set=support_set, 
                             similarity=similarity).to(device)
    
    predictor.train()

    # Only learn parameters W and b in the Softmax classifier
    optimizer = Adam([predictor.softmax_classifier_w, predictor.softmax_classifier_b], lr=learning_rate)

    criterion = nn.CrossEntropyLoss()
    start_time = time.time()
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

        if entropy_regularization == 'shannon':
            # Calculate entropy regularization as cross-entropy of softmaxed logits with themselves
            softmaxed_logits = torch.softmax(logits, dim=1)
            entropy_regularize = -(softmaxed_logits * torch.log(softmaxed_logits + 1e-6)).sum(dim=1).mean()
            loss += entropy_regularize
        elif entropy_regularization == 'renyi':
             # Calculate softmax probabilities of logits
            softmaxed_logits = torch.softmax(logits, dim=1)
            # Prevent taking log of zero by adding a small constant
            renyi_entropy = torch.log((softmaxed_logits ** alpha).sum(dim=1) + 1e-6) / (alpha - 1)
            entropy_regularize = -renyi_entropy.mean()
            loss += entropy_regularize

        if (epoch + 1) % 20 == 0:
            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end_time = time.time()

    # Save trained parameters
    if is_baseline:
        torch.save(predictor.state_dict(), f'models/baseline_predictor_{entropy_regularization}_sim_{similarity}.pth')
    else:
        torch.save(predictor.state_dict(), f'models/predictor_{entropy_regularization}_sim_{similarity}.pth')

    return predictor, end_time - start_time



def load_model(model_path, embedding_dim=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = FSCEmbNet(embedding_dim=embedding_dim).to(device=device)
    model.load_state_dict(torch.load(model_path))
    return model


def test(predictor, test_set):
    predictor.eval()
    device = next(predictor.parameters()).device

    total_test_samples_num = 0
    total_right_samples_num = 0
    all_labels = []
    all_preds = []
    all_probs = []  # Store probabilities for ROC calculation
    inference_times = []

    print('============================================================')

    with torch.no_grad():
        for class_id, this_class_data in test_set.items():
            start_time = time.time()

            this_class_samples_num = len(this_class_data)
            total_test_samples_num += this_class_samples_num

            this_class_data = torch.stack(this_class_data).to(device)
            predict_this_class = predictor(this_class_data)
            elapsed_time = time.time() - start_time
            inference_time = elapsed_time / this_class_samples_num
            inference_times.append(inference_time)

            predict_probs = torch.nn.functional.softmax(predict_this_class, dim=1).cpu().numpy()  # Get probabilities
            predict_label = np.argmax(predict_probs, axis=1)

            all_labels.extend([class_id] * this_class_samples_num)
            all_preds.extend(predict_label.tolist())
            all_probs.extend(predict_probs)

            this_right_samples_num = np.sum(predict_label == class_id)
            total_right_samples_num += this_right_samples_num
            accuracy = this_right_samples_num / this_class_samples_num
            print(f'Label:{class_id}, Samples:{this_class_samples_num}, Correct:{this_right_samples_num}, Accuracy:{accuracy:.2f}, Avg. inference time/image:{inference_time:.4f}s')

    all_probs = np.array(all_probs)
    all_labels_binary = label_binarize(all_labels, classes=np.unique(all_labels))

    # Calculate and plot averaged ROC curve
    mean_fpr = np.linspace(0, 1, 100)  # Common set of false positive rates
    tprs = []
    aucs = []

    for i, class_id in enumerate(np.unique(all_labels)):
        fpr, tpr, _ = roc_curve(all_labels_binary[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    overall_accuracy = total_right_samples_num / total_test_samples_num
    macro_recall = recall_score(all_labels, all_preds, average='macro')
    macro_precision = precision_score(all_labels, all_preds, average='macro')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    avg_inference_time = np.mean(inference_times)

    print(f'Total samples: {total_test_samples_num}, Total correct: {total_right_samples_num}')
    print(f'Overall Accuracy: {overall_accuracy:.2f}')
    print(f'Macro Recall: {macro_recall:.2f}')
    print(f'Macro Precision: {macro_precision:.2f}')
    print(f'Macro F1 Score: {macro_f1:.2f}')
    print(f'Average Inference Time per Image: {avg_inference_time:.4f}s')

    return mean_fpr, mean_tpr, mean_auc

def save_roc_data(fpr, tpr, roc_auc, filename):
    with open(filename, 'wb') as f:
        pickle.dump({'mean_fpr': fpr, 'mean_tpr': tpr, 'mean_auc': roc_auc}, f)


def main(support_path, test_path, emb_model_path, sim, entropy_regularize):
    # load support
    with open(support_path, 'rb') as file:
        support_set = pickle.load(file)
    # load test
    with open(test_path, 'rb') as file:
        test_set = pickle.load(file)
    
    predictor_epochs = 200
    predictor_learning_rate = 0.0005

    emb_model = load_model(emb_model_path, embedding_dim=128)

    predictor, dur = train_predictor(emb_model, support_set, epochs=predictor_epochs, 
                                learning_rate=predictor_learning_rate,
                                similarity=sim,
                                entropy_regularization=entropy_regularize)

    print(f"Fine tune time with single GPU: {dur:.4f}s")
    # Calculate the number of parameters
    total_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")

    # Perform the test using the loaded predictor and the prepared test set
    print(f"Test results when Similarity function is {sim} & entropy_regularize is {entropy_regularize}")
    fpr, tpr, roc_auc = test(predictor, test_set)
    save_roc_data(fpr, tpr, roc_auc, 'res18_triplet_roc.pkl')



# exmaple commands:
# python res.py -sp ./support_set.pkl -tp ./test_set.pkl -m models/emb_model.pth -s cosine
# python res.py -sp ./support_set.pkl -tp ./test_set.pkl -m models/emb_model.pth -s l1 -e renyi
# python res.py -sp ./support_set.pkl -tp ./test_set.pkl -m models/emb_model.pth -s l2 -e shannon
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tuning & Test")
    parser.add_argument('-sp', '--support_path', type=str, required=True, help='Support set path')
    parser.add_argument('-tp', '--test_path', type=str, required=True, help='Test set path')
    parser.add_argument('-m', '--emb_model_path', type=str, required=True, help='Path to the embedding model file')
    parser.add_argument('-s', '--sim', type=str, required=True, help='Similarity function to use')
    parser.add_argument('-e', '--entropy_regularize', type=str, default=None, required=False, help='Entropy regularization')

    args = parser.parse_args()
    main(args.support_path, args.test_path, args.emb_model_path, args.sim, args.entropy_regularize)
