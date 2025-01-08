import os
import sys
import time
import random
import numpy as np
from torch.utils.data import Subset
import torch
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataloader import DataLoaderFactory  
from src.model.model import LRP_NN, LRP_CNN  
from utils import save_results_to_pdf_experiment2

class Logger:
    def __init__(self, filename="log.txt"):
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory) 
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect sys.stdout to an instance of Logger
timestamp = time.strftime("%Y%m%d-%H%M%S")  
logger= Logger(f"experiment 2 results/log experiment 2 {timestamp}")
sys.stdout = logger


# Function to set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

 
def train_and_evaluate(seed, model_class, train_loader, test_loader, subset_indices, epochs=4, base_learning_rate=0.001,
                       use_lrp=True, explain_layers=['fc1', 'fc2', 'fc3'], factor=1, 
                       apply_lrp_to ="all", percentage=100):

    # Adjust window_size and batch_patience based on the percentage
    if percentage >= 50:
        window_size = 10
        batch_patience = 40
    elif percentage >= 10:
        window_size = 8
        batch_patience = 20
    else:
        window_size = 10
        batch_patience = 6

    reduced_train_loader = torch.utils.data.DataLoader(
        Subset(train_loader.dataset, subset_indices), batch_size=train_loader.batch_size, shuffle=False
    )
    
    set_seed(seed)
    if model_class == LRP_NN:
        set_seed(seed)
        start_time = time.time()
        model = model_class(factor=factor)
        final_acc = model.train_model_early_stopping(
                reduced_train_loader, test_loader, epochs=epochs, base_learning_rate=base_learning_rate, use_lrp=use_lrp,
                explain_layers=explain_layers, window_size=window_size, batch_patience=batch_patience
        )
        end_time = time.time()  
        train_time = end_time - start_time 

    elif model_class == LRP_CNN:
        set_seed(seed)
        start_time = time.time()
        model = model_class(factor=factor)
        final_acc = model.train_model_early_stopping(
            reduced_train_loader, test_loader, epochs=epochs, base_learning_rate=base_learning_rate, use_lrp=use_lrp,
            window_size=window_size, batch_patience=batch_patience, apply_lrp_to = apply_lrp_to
        )
        end_time = time.time()  
        train_time = end_time - start_time 

    return final_acc, train_time


def run_experiment(seed, model_class, train_loader, test_loader, percentages, epochs=3, base_learning_rate=0.001, explain_layers=['fc1', 'fc2', 'fc3'], num_repeats=1, factor = 1, apply_lrp_to = "all"):
    stats_lrp = []
    stats_no_lrp = []
    time_lrp = []
    time_no_lrp = []

    for percentage in percentages:
        print(f"Training with {percentage}% of training data...")

        total_train_size = len(train_loader.dataset)
        train_subset_size = int(total_train_size * (percentage / 100))
        
        lrp_accuracies = []
        no_lrp_accuracies = []
        lrp_times = []
        no_lrp_times = []
  
        for repeat in range(num_repeats):
            print(f"Repeat {repeat + 1}/{num_repeats} for {percentage}% data")

            seed = seed + repeat 
            print(f"the seed used in this run " , seed)
            indices = list(range(total_train_size))
            set_seed(seed)
            np.random.shuffle(indices)
            subset_indices = indices[:train_subset_size]

            lrp_accuracy , lrp_time= train_and_evaluate(
                seed, model_class, train_loader, test_loader, subset_indices, epochs, base_learning_rate, 
                use_lrp=True, explain_layers=explain_layers, factor=factor,
                 apply_lrp_to=apply_lrp_to, percentage=percentage
            )
            lrp_accuracies.append(lrp_accuracy)
            lrp_times.append(lrp_time)

            
            no_lrp_accuracy , no_lrp_time= train_and_evaluate(
                seed, model_class, train_loader, test_loader, subset_indices, epochs, base_learning_rate, use_lrp=False, apply_lrp_to="none", percentage=percentage
            )
            no_lrp_accuracies.append(no_lrp_accuracy)
            no_lrp_times.append(no_lrp_time)
        
        stats_lrp.append(np.mean(lrp_accuracies))
        stats_no_lrp.append(np.mean(no_lrp_accuracies))
        time_lrp.append(np.mean(lrp_times))
        time_no_lrp.append(np.mean(no_lrp_times))
      
    return stats_lrp, stats_no_lrp, time_lrp, time_no_lrp

def load_config(config_path="Layerwise-Relevance-Propagation-Implementation/experiments/config.yaml"):
    """Load YAML configuration."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


if __name__ == "__main__":

    config = load_config()

    if config['use_config']:
        percentages1 = config['percentages']['percentages1']
        percentages2 = config['percentages']['percentages2']
        model_type = eval(config['model_type']) 
        apply_lrp_to = config['apply_lrp_to']
        seed = config['seed']
        epochs = config['epochs']
        num_repeats = config['num_repeats']
        factors = config['factors']
    else:

        percentages1 = [ 100, 90, 80, 70, 60, 50, 40,30,20,10,5]
        percentages2 =  [ 30, 20,17.5, 15, 12.5, 10, 7.5, 5, 2.5, 1, 0.5, 0.1]

        model_type = LRP_CNN   #LRP_NN
       
        factors = [0.1]
        i=  1
        apply_lrp_to =  "linear"  #"all"
        num_repeats = 1
        epochs = 4
    
    percentages = [percentages1]
    
    for factor in factors:
        for percentage in percentages:
            explain_layers = ['fc1', 'fc2', 'fc3']
            base_learning_rate = 0.001

            if model_type == LRP_CNN:
                train_loader, test_loader = DataLoaderFactory.load_data(dataset_name="CIFAR10", batch_size=64)
            elif model_type == LRP_NN:
                train_loader, test_loader = DataLoaderFactory.load_data(dataset_name="MNIST", batch_size=64)
            
            stats_lrp, stats_no_lrp , time_lrp, time_no_lrp = run_experiment(
                seed, model_type, train_loader, test_loader, percentage, epochs=epochs, base_learning_rate=base_learning_rate, explain_layers=explain_layers, num_repeats=num_repeats, factor=factor, apply_lrp_to = apply_lrp_to
            )
            print("lrp time")
            print(time_lrp), 
            print("\n time_no_lrp")
            print(time_no_lrp)
            save_results_to_pdf_experiment2(i ,percentage, stats_lrp, stats_no_lrp, epochs, base_learning_rate, num_repeats, model_type = model_type, apply_lrp_to = apply_lrp_to, factor = factor, time_lrp = time_lrp, time_no_lrp= time_no_lrp, pdf_filename='experiment_results.pdf')
            i = i+1
   

os.system('notify-send "Experiment Complete" "The experiment has finished running!"')

sys.stdout = sys.__stdout__  
logger.log.close() 