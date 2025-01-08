import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_optimizer as torch_opt

from .linearlrp import LinearLRPLayer
from .convlrp import Conv2dLRPLayer
from .maxPool2dLRP import MaxPool2dLRPLayer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CNN(nn.Module): 
    def __init__(self):
        super(CNN, self).__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 784)  
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  
        return x
    


class LRP_CNN(nn.Module):
    def __init__(self, factor=0.01):
        super(LRP_CNN, self).__init__()
        self.factor = torch.tensor(factor, requires_grad=False)
        self.conv1 = Conv2dLRPLayer(3, 64, kernel_size=3, padding=1)
        self.conv2 = Conv2dLRPLayer(64, 64, kernel_size=3, padding=1)
        self.maxpool1 = MaxPool2dLRPLayer(kernel_size=2, stride=2)

        self.conv3 = Conv2dLRPLayer(64, 128, kernel_size=3, padding=1)
        self.conv4 = Conv2dLRPLayer(128, 128, kernel_size=3, padding=1)
        self.maxpool2 = MaxPool2dLRPLayer(kernel_size=2, stride=2)

        self.conv5 = Conv2dLRPLayer(128, 256, kernel_size=3, padding=1)
        self.maxpool3 = MaxPool2dLRPLayer(kernel_size=2, stride=2)

        self.fc1 = LinearLRPLayer(256 * 4 * 4, 1024)
        self.fc2 = LinearLRPLayer(1024, 512)
        self.fc3 = LinearLRPLayer(512, 10)

    def forward(self, x, explain=False, lrp = False, rule='lrp0', apply_lrp_to="all"):
        
        factor_tensor = self.factor.to(x.device)
        # Convolution and Max Pooling Layers
        apply_lrp_conv = (apply_lrp_to == "all")
        x = F.relu(self.conv1(x, explain=explain, lrp=apply_lrp_conv, rule=rule,factor=factor_tensor, is_output_layer = False))
        x = F.relu(self.conv2(x, explain=explain, lrp=apply_lrp_conv, rule=rule,factor=factor_tensor, is_output_layer = False))
        x = self.maxpool1(x, explain=explain, lrp=apply_lrp_conv, rule=rule, is_output_layer = False)

        x = F.relu(self.conv3(x, explain=explain, lrp=apply_lrp_conv, rule=rule,factor=factor_tensor, is_output_layer = False))
        x = F.relu(self.conv4(x, explain=explain, lrp=apply_lrp_conv, rule=rule,factor=factor_tensor, is_output_layer = False))
        x = self.maxpool2(x, explain=explain, lrp=apply_lrp_conv, rule=rule, is_output_layer = False)

        x = F.relu(self.conv5(x, explain=explain, lrp=apply_lrp_conv, rule=rule,factor=factor_tensor, is_output_layer = False))
        x = self.maxpool3(x, explain=explain, lrp=apply_lrp_conv, rule=rule, is_output_layer = False)

        x = x.view(x.size(0), -1)

        apply_lrp_linear = (apply_lrp_to == "all" or apply_lrp_to == "linear")
        connect_not_to_cnn = apply_lrp_to == "linear"
        x = F.relu(self.fc1(x, explain=explain, lrp=apply_lrp_linear, rule=rule, is_output_layer = connect_not_to_cnn, factor=factor_tensor))
        x = F.relu(self.fc2(x, explain=explain, lrp=apply_lrp_linear, rule=rule, is_output_layer = False, factor=factor_tensor,))
        x = self.fc3(x, explain=explain, lrp=apply_lrp_linear, rule=rule,is_output_layer=True, factor=factor_tensor,)

        return x


    def train_model(self, train_loader, test_loader, epochs=5, base_learning_rate=0.001,  explain = False, use_lrp=False, rule='lrp0', apply_lrp_to="all"):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=base_learning_rate)
        total_start_time = time.time()
        epoch_times = []   
        epochs_accuracy = []    
        for epoch in range(epochs):
            epoch_start_time = time.time()
            self.train()
            running_loss = 0.0

            for images, labels in train_loader:
                optimizer.zero_grad()
                
                outputs = self(images, explain=explain, lrp=use_lrp, rule=rule, apply_lrp_to=apply_lrp_to)
                loss = criterion(outputs, labels)
                loss.backward() 
                optimizer.step()  
                running_loss += loss.item()

            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}')
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            final_accuracy = self.evaluate_model(test_loader)
            epochs_accuracy.append(final_accuracy)  
            print(f"Epoch [{epoch + 1}] finished with accuracy: {final_accuracy:.2f}%")

        total_training_time = time.time() - total_start_time
        average_epoch_time = sum(epoch_times) / len(epoch_times)
        final_accuracy = epochs_accuracy[-1]  # Accuracy of the final epoch

        return total_training_time, average_epoch_time, final_accuracy, epochs_accuracy
    
    def train_model_early_stopping(self, train_loader, val_loader, epochs=4, base_learning_rate=0.001, explain=False, use_lrp=False, rule='lrp0', apply_lrp_to="all", batch_patience=5, model_path="best_model.pth", window_size=10):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=base_learning_rate)
        best_val_accuracy = 0.0  
        patience_counter = 0  
        best_model_state = None  

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0 
            # only apply early stopping on the last two layers
            is_last_two_epochs = (epoch >= epochs - 1) 

            for i, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = self(images, explain=explain, lrp=use_lrp, rule=rule, apply_lrp_to=apply_lrp_to)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()  
            
                if is_last_two_epochs and (i + 1) % window_size == 0:
                    self.eval()  
                    val_loss = 0.0
                    correct = 0
                    total = 0

                    with torch.no_grad():
                        for val_images, val_labels in val_loader:
                            val_outputs = self(val_images, explain=explain, lrp=use_lrp, rule=rule, apply_lrp_to=apply_lrp_to)
                            val_loss += criterion(val_outputs, val_labels).item()
                            _, predicted = torch.max(val_outputs, 1)
                            total += val_labels.size(0)
                            correct += (predicted == val_labels).sum().item()
                    
                    # Calculate average validation loss and accuracy
                    val_loss /= len(val_loader)
                    val_accuracy = 100 * correct / total

                    print(f"Epoch [{epoch + 1}], Batch [{i + 1}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

                    # Early stopping logic: check if validation accuracy improved
                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy  # Update the best accuracy
                        patience_counter = 0  # Reset patience counter if there's improvement
                        best_model_state = self.state_dict() # Save the best model state
                        torch.save(best_model_state, model_path)
                    else:
                        patience_counter += 1  # Increment patience counter if no improvement

                    # Trigger early stopping if no improvement within `batch_patience` windows
                    if patience_counter >= batch_patience:
                        print(f"Early stopping triggered at epoch {epoch + 1}, batch {i + 1}")
                        break  # Exit the batch loop early if stopping criteria met

                    self.train()  

            # Stop the epoch loop if early stopping was triggered in the last two epochs
            if is_last_two_epochs and patience_counter >= batch_patience:
                break 
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}')
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
        print("Best Validation Accuracy:", best_val_accuracy)
        return best_val_accuracy

    def evaluate_model(model, test_loader, flatten=False):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                if flatten:
                    images = images.view(images.size(0), -1)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy:.2f}%')
        return accuracy


class LRP_NN(nn.Module):
    def __init__(self, factor=1):
        super(LRP_NN, self).__init__()
        self.factor = torch.tensor(factor, requires_grad=False)
        self.fc1 = LinearLRPLayer(784, 300)
        self.fc2 = LinearLRPLayer(300, 100)
        self.fc3 = LinearLRPLayer(100, 10)
        

    def forward(self, x, explain=False, lrp= False, rule='lrp0', explain_layers=None):
        factor_tensor = self.factor.to(x.device)
        x = x.view(-1, 784)  
        x = F.relu(self.fc1(x, explain=explain, lrp=lrp, rule=rule, is_output_layer=False, factor=factor_tensor))
        x = F.relu(self.fc2(x, explain=explain, lrp=lrp, rule=rule, is_output_layer=False, factor=factor_tensor))
        x = self.fc3(x, explain=explain, lrp=lrp, rule=rule, is_output_layer=True, factor=factor_tensor)
        return x

    def train_model(self, train_loader, test_loader, epochs=5, base_learning_rate=0.001, explain = False, use_lrp=False, rule='lrp0', explain_layers=None):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=base_learning_rate)

        total_start_time = time.time()
        epoch_times = []  
        epochs_accuracy = []     
        for epoch in range(epochs):
            epoch_start_time = time.time()
            self.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images = images.view(images.size(0), -1) 
                optimizer.zero_grad()
                outputs = self(images, explain=explain, lrp = use_lrp, rule=rule, explain_layers=explain_layers) 
                loss = criterion(outputs, labels)
                loss.backward() 
                optimizer.step()  
                running_loss += loss.item()

            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}')

            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            final_accuracy = self.evaluate_model(test_loader)
            epochs_accuracy.append(final_accuracy)  # 
            print(f"Epoch [{epoch + 1}] finished with accuracy: {final_accuracy:.2f}%")

        total_training_time = time.time() - total_start_time
        average_epoch_time = sum(epoch_times) / len(epoch_times)
        final_accuracy = epochs_accuracy[-1] 

        return total_training_time, average_epoch_time, final_accuracy, epochs_accuracy
    
    def train_model_early_stopping(self, train_loader, val_loader, epochs=4, base_learning_rate=0.001, explain=False, use_lrp=False, rule='lrp0', explain_layers=None, batch_patience=5, model_path="best_model.pth", window_size=10):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=base_learning_rate)

        best_val_accuracy = 0.0
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            is_last_two_epochs = (epoch >= epochs - 2) 

            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.view(images.size(0), -1) 

                optimizer.zero_grad()
                outputs = self(images, explain=explain, lrp=use_lrp, rule=rule, explain_layers=explain_layers)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if is_last_two_epochs and (batch_idx + 1) % window_size == 0:
                    self.eval()
                    val_accuracy = self.evaluate_model(val_loader)
                    print(f"Epoch [{epoch + 1}], Batch [{batch_idx + 1}], Validation Accuracy: {val_accuracy:.2f}%")

                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        patience_counter = 0
                        best_model_state = self.state_dict()
                        torch.save(best_model_state, model_path)
                    else:
                        patience_counter += 1

                    if patience_counter >= batch_patience:
                        print(f"Early stopping triggered at epoch {epoch + 1}, batch {batch_idx + 1}")
                        break  

                    self.train()

            # Stop epoch loop if early stopping was triggered during the last two epochs
            if is_last_two_epochs and patience_counter >= batch_patience:
                break

            if not is_last_two_epochs:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}')

        if best_model_state is not None:
            self.load_state_dict(best_model_state)
        print("Best Validation Accuracy:", best_val_accuracy)
        return best_val_accuracy



    def evaluate_model(self, test_loader):
        self.eval()  
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.view(images.size(0), -1)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy:.2f}%')
        return accuracy




