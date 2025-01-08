import torch
from torch.utils.data import Subset
import random
from dataloader import DataLoaderFactory
from model.model import LRP_NN
import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import matplotlib.pyplot as plt
import os

def plot_input_image(image):
    image = image.view(28, 28).detach().cpu().numpy()
    plt.imshow(image, cmap='gray')
    plt.title("Input Image")
    plt.axis('off')
    plt.show()

def print_model_parameters(model, model_name="Model"):
            for name, param in model.named_parameters():
                print(f"{model_name} - {name}: {param.data.mean():.4f}, {param.data.std():.4f}")

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
    relevance_scores = (relevance_scores - relevance_scores.min()) / (relevance_scores.max() - relevance_scores.min())
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
    dataset = "MNIST"
    explain_layers = ['fc1', 'fc2', 'fc3']
    
    percentage = None
    logging = False

    use_weightened_samples = None

    print("Choose a training method:")
    print("1. Standard Training")
    print("2. LRP-based Training")

    choice = input("Enter the number of your choice (1/2): ")
    use_lrp = (choice == '2')

    train_loader, test_loader = DataLoaderFactory.load_data(dataset_name=dataset, batch_size=64)
    factor = float(input("Enter the factor to scale relevance cache during training (e.g., 0.1, 0.5, 1): "))

    set_seed(42)
    if logging is None:
        logging = str(input("Do you want to log the results into the logs folder? (yes/no): ")) 
        model = LRP_NN(factor=factor)
    else: 
        model = LRP_NN(factor=factor)

    print_model_parameters(model)
    if percentage is None:
        percentage = float(input("Enter the percentage of training data to use (e.g., 1 for 100%, 0.5 for 50%): "))

    if use_weightened_samples is None:
        use_weightened_samples = str(input("Do you want to the sample to be weightened by their lrp score? (yes/no): "))
        if use_weightened_samples == "yes" or use_weightened_samples =="y":
            use_weightened_samples = True
        elif use_weightened_samples =="no" or "n": 
            use_weightened_samples = False


    train_dataset = train_loader.dataset
    train_size = int(len(train_dataset) * percentage)
    indices = list(range(len(train_dataset)))
    random.shuffle(indices) 
    subset_indices = indices[:train_size]
    
    reduced_train_loader = torch.utils.data.DataLoader(
        Subset(train_dataset, subset_indices), batch_size=train_loader.batch_size, shuffle=True
    )
   
    total_time, avg_epoch_time, final_acc,_ = model.train_model(
        reduced_train_loader, test_loader, epochs=3, base_learning_rate=0.001,use_lrp=use_lrp, explain_layers=explain_layers
    )
    
    print("\n-------- Stats --------")
    print(f"Total training time: {total_time:.4f} seconds")
    print(f"Average time per epoch: {avg_epoch_time:.4f} seconds")
    print(f"Final accuracy: {final_acc:.2f}%")
    print("------------------------")

    # Select a single image from the test set
    for images, labels in test_loader:
        input_image = images[2].unsqueeze(0)  
        input_image.requires_grad_() 
        break

    plot_input_image(input_image)
    output_scores = model(input_image, explain=False, lrp=True)
    predicted_class = output_scores.argmax(dim=1).item()

    relevance_scores = model(input_image, explain=True, rule='lrp0')

    relevance_scores_for_class = relevance_scores[:, predicted_class]
    relevance_scores_for_class.sum().backward()  

    relevance_at_input = input_image.grad

    relevance_at_input = relevance_at_input.view(28, 28).cpu().detach().numpy()

    plot_relevance_scores(relevance_at_input, input_image)

