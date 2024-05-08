import pickle
import matplotlib.pyplot as plt
import numpy as np

def load_roc_metrics(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_roc_curves(model_files, model_labels, save_path='roc_comparison.png'):
    plt.figure(figsize=(10, 8))
    for model_file, label in zip(model_files, model_labels):
        data = load_roc_metrics(model_file)
        plt.plot(data['mean_fpr'], data['mean_tpr'], label=f'{label} (Macro AUC = {data["mean_auc"]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')  # Add a diagonal dashed line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Macro ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.savefig(save_path)  # Save the plot to a file
    plt.close()  # Close the plot to free up memory

# Example usage
model_files = ['base_roc.pkl','res18_triplet_roc.pkl','res18_cos_roc.pkl', 'res18_cos_roc_shannon.pkl','res18_cos_roc_renyi.pkl']
model_labels = ['Baseline', 'FSL(Triplet_loss)','FSL(Triplet_similarity)', 'With Shannon','With Renyi']
plot_roc_curves(model_files, model_labels, 'roc_comparison.png')
