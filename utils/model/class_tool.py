import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch_geometric.nn import GCNConv, GATConv
import matplotlib.pyplot as plt
import numpy as np


# Define the Graph Autoencoder Model
class GAEModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(GAEModel, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, 8)
        self.fc = nn.Linear(8, num_classes)  # Output layer with num_classes

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.fc(z)  # Output logits


class GATModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, 16)
        self.conv2 = GATConv(16, 8)
        self.fc = nn.Linear(8, num_classes)  # Output layer with num_classes

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.fc(z)  # Output logits


class GCNModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, 8)
        self.fc = nn.Linear(8, num_classes)  # Output layer with num_classes

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.fc(z)  # Output logits


# Common loss functions
def bce_loss(z, pos_edge_index, neg_edge_index):
    pos_edge_weight = torch.sigmoid((z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1))
    neg_edge_weight = torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1))
    pos_edge_weight = torch.clamp(pos_edge_weight, min=1e-10, max=1 - 1e-10)
    neg_edge_weight = torch.clamp(neg_edge_weight, min=1e-10, max=1 - 1e-10)
    return -torch.mean(torch.log(pos_edge_weight) + torch.log(1 - neg_edge_weight))


def mse_loss(z, pos_edge_index, neg_edge_index):
    pos_loss = torch.mean((z[pos_edge_index[0]] - z[pos_edge_index[1]]) ** 2)
    neg_loss = torch.mean((z[neg_edge_index[0]] - z[neg_edge_index[1]]) ** 2)
    return pos_loss + neg_loss


# Train the model
def train_gae(data, save_path, epoch, lr,  num_classes, model_class=GAEModel, loss_fn=bce_loss):
    model = model_class(data.x.size(1), num_classes).to(data.x.device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epoch):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)

        # Get positive and negative samples
        pos_edge_index = data.edge_index
        neg_edge_index = data.edge_index[:, torch.randperm(data.edge_index.size(1))[:pos_edge_index.size(1)]]

        # Compute loss
        loss = loss_fn(logits, pos_edge_index, neg_edge_index)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 20 == 0:
            print(f'Epoch {epoch}: Train Loss {loss.item():.4f}')

    # Save model parameters
    torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))

    # Plot loss function graph
    plot_loss(losses, save_path)


def plot_loss(losses, save_path):
    smoothed_losses = np.convolve(losses, np.ones(10) / 10, mode='valid')

    plt.figure(figsize=(10, 5))
    plt.plot(smoothed_losses, 'b-', label='Smoothed Loss')
    plt.plot(losses, 'ro', label='Loss Values')

    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_path, 'loss_plot.png'))


# Use the trained model for label prediction
def predict_labels(data, model_class, model_path, num_classes, output_path):
    print("Feature matrix shape during prediction:", data.x.shape)
    model = model_class(data.x.size(1), num_classes).to(data.x.device)

    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    except RuntimeError as e:
        print(f"Error loading model state_dict: {e}")
        return
    except FileNotFoundError as e:
        print(f"Model file not found: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading the model: {e}")
        return

    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)

    # Apply softmax to compute probabilities
    probs = torch.softmax(logits, dim=-1)
    predicted_labels = torch.argmax(probs, dim=-1)

    df_labels = pd.DataFrame(predicted_labels.cpu().numpy(), columns=['Label'])
    df_labels.to_csv(os.path.join(output_path, 'predicted_labels.csv'), index=False)
    print(f"Predicted labels saved to {output_path}")
