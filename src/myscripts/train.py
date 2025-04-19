from torch.utils.data import DataLoader, TensorDataset
from src.myscripts.model import EarlyStopping
import torch.nn as nn
import torch

class ModelTrainer:
    def __init__(self, model, device):
        self.device = device
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.result_model = None
        self.train_losses = []
        self.val_losses = []

    def train_model(self, train_dataset, val_dataset, epochs=10, early_stopping_rounds=3):
        if not isinstance(train_dataset, TensorDataset) or not isinstance(val_dataset, TensorDataset):
            raise TypeError("train_dataset and val_dataset must be TensorDataset objects!")

        # Trenowanie
        early_stopping = EarlyStopping(patience=early_stopping_rounds, verbose=True)
        # Listy do przechowywania strat


        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        # Trenowanie modelu
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0
            correct = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels)

            epoch_loss = running_loss / len(train_loader)
            acc = correct.item() / len(train_loader.dataset)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {acc:.4f}")

            # Dodanie straty treningowej do listy
            self.train_losses.append(epoch_loss)

            # Ewaluacja na zbiorze walidacyjnym
            self.model.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    correct += torch.sum(preds == labels)

            val_loss /= len(val_loader)  # średnia strata na zbiorze walidacyjnym
            val_accuracy = correct.item() / len(val_loader.dataset)
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

            # Dodanie straty walidacyjnej do listy
            self.val_losses.append(val_loss)

            # Sprawdzamy, czy należy zatrzymać trening
            early_stopping(val_loss, self.model)

            # Jeśli early stopping mówi "stop", przerywamy trening
            if early_stopping.early_stop:
                print("Early stopping triggered. Stopping training.")
                break
        self.result_model=early_stopping.load_best_model(self.model)



    def get_trained_model(self):
        if self.result_model is None:
            raise ValueError("Model has not been trained yet. Please run \"train_model\" method!")
        return self.result_model

    def get_loss_data(self):
        if self.result_model is None:
            raise ValueError("Model has not been trained yet. Cannot aquire loss data Please run \"train_model\" method!")
        return self.train_losses, self.val_losses
