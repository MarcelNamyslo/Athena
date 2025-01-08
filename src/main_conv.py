
import torch
from torch.utils.data import Subset
import random
from dataloader import DataLoaderFactory
from model.model import LRP_CNN

import matplotlib.pyplot as plt
import os


def plot_input_image(image):
    if image.dim() == 4:
        image = image[0] 
    image = image.permute(1, 2, 0).detach().cpu().numpy()
    if image.shape[-1] == 1:
        image = image.squeeze(-1)

    # Plot the image
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def save_model_in_folder(model, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    existing_models = [f for f in os.listdir(folder_path) if f.startswith("model") and f.endswith(".pth")]
    model_number = len(existing_models) + 1
    
    model_name = f"model{model_number}.pth"
    model_path = os.path.join(folder_path, model_name)
    
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_name} in {folder_path}")

def plot_relevance_scores(relevance_scores, input_image):

    if input_image.dim() == 4: 
        input_image = input_image.squeeze(0)  
    input_image = input_image.permute(1, 2, 0).detach().cpu().numpy()
    relevance_scores = (relevance_scores - relevance_scores.min()) / (relevance_scores.max() - relevance_scores.min())

    plt.imshow(input_image, cmap='gray' if input_image.shape[-1] == 1 else None, alpha=0.6)
    heatmap = plt.imshow(relevance_scores, cmap='jet', alpha=0.4)
    cbar = plt.colorbar(heatmap)
    cbar.set_label('Relevance Score')
    plt.title("Relevance Scores Overlay")
    plt.axis('off')
    plt.show()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":

    percentage = None
    logging = None

    factor = float(input("Enter the factor to scale relevance score during training (e.g., 0.1, 0.5, 1): "))
    apply_lrp_to = input("Apply LRP to which layers? Enter 'all' or 'linear': ").strip().lower()

    train_loader, test_loader = DataLoaderFactory.load_data(dataset_name="CIFAR10", batch_size=64)
    set_seed(42)
    model = LRP_CNN(factor = factor)

    if percentage is None:
        percentage = float(input("Enter the percentage of training data to use (e.g., 1 for 100%, 0.5 for 50%): "))
    train_dataset = train_loader.dataset
    train_size = int(len(train_dataset) * percentage)
    indices = list(range(len(train_dataset)))
    random.shuffle(indices) 
    subset_indices = indices[:train_size]
    
    reduced_train_loader = torch.utils.data.DataLoader(
        Subset(train_dataset, subset_indices), batch_size=train_loader.batch_size, shuffle=True
    )

    print("Choose a training method:")
    print("1. Standard Training")
    print("2. LRP-based Training")

    choice = input("Enter the number of your choice (1/2): ")
    use_lrp = (choice == '2')

    total_time, avg_epoch_time, final_acc = model.train_model(
        train_loader, test_loader, epochs=3, base_learning_rate=0.001, use_lrp=True, rule='lrp0', apply_lrp_to=apply_lrp_to
    )

    print("\n-------- Stats --------")
    print(f"Total training time: {total_time:.4f} seconds")
    print(f"Average time per epoch: {avg_epoch_time:.4f} seconds")
    print(f"Final accuracy: {final_acc:.2f}%")
    print("------------------------")

    for images, labels in test_loader:
        input_image = images[2].unsqueeze(0)  
        input_image.requires_grad_() 
        break

    plot_input_image(input_image)


    output_scores = model(input_image, explain=False, lrp=False)
    predicted_class = output_scores.argmax(dim=1).item()

  
    relevance_scores = model(input_image, explain=True, rule='lrp0')


    relevance_scores_for_class = relevance_scores[:, predicted_class]
    relevance_scores_for_class.sum().backward()  

    relevance_at_input = input_image.grad

 
    if input_image.size(1) == 1:  
        relevance_at_input = relevance_at_input.view(28, 28).cpu().detach().numpy()
    else: 
        relevance_at_input = relevance_at_input.view(3, 32, 32).permute(1, 2, 0).cpu().detach().numpy() 
    plot_relevance_scores(relevance_at_input, input_image)

