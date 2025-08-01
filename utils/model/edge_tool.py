import os
from datetime import date
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, ChebConv, GINConv, GATConv, GCNConv
import torch.optim as optim
from scipy.sparse import coo_matrix, lil_matrix

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def add_self_loops(edge_index, num_nodes):
    loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index_with_loops = torch.cat([edge_index, loop_index], dim=1)
    return edge_index_with_loops


class DynamicGCN(torch.nn.Module):
    def __init__(self, conv_channels):
        super(DynamicGCN, self).__init__()
        self.layers = torch.nn.ModuleList([
            GCNConv(conv_channels[i], conv_channels[i + 1])
            for i in range(len(conv_channels) - 1)
        ])

    def forward(self, x, edge_index):
        # 检查 edge_index 的格式
        if edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")

        # To add self loops, use the add_self_loops function you provided
        edge_index_with_loops = add_self_loops(edge_index, num_nodes=x.size(0))

        # Iterate through each layer of the GCN
        for layer in self.layers:
            x = layer(x, edge_index_with_loops)
            x = F.elu(x)
            x = F.dropout(x, p=0.3, training=self.training)

        return x


class DynamicGAT(torch.nn.Module):
    def __init__(self, conv_channels):
        super(DynamicGAT, self).__init__()
        self.layers = torch.nn.ModuleList([
            GATConv(conv_channels[i], conv_channels[i + 1], heads=1, dropout=0.3)
            for i in range(len(conv_channels) - 1)
        ])

    def forward(self, x, edge_index):
        # Check the format of edge_index
        # Make sure the edge_index dimension is [2, num_edges].
        if edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")

        # Add self-loop
        edge_index_with_loops = add_self_loops(edge_index, num_nodes=x.size(0))

        for layer in self.layers:
            x = F.dropout(x, p=0.3, training=self.training)
            x = layer(x, edge_index_with_loops)
            x = F.elu(x)
        return x


class DynamicGraphSAGE(torch.nn.Module):
    def __init__(self, conv_channels):
        super(DynamicGraphSAGE, self).__init__()
        self.layers = torch.nn.ModuleList([
            SAGEConv(conv_channels[i], conv_channels[i + 1])
            for i in range(len(conv_channels) - 1)
        ])

    def forward(self, x, edge_index):
        # Convert sparse matrices to dense matrices
        if isinstance(edge_index, torch.sparse.Tensor):
            edge_index = edge_index.to_dense()

        for layer in self.layers:
            x = F.dropout(x, p=0.3, training=self.training)
            x = layer(x, edge_index)
            x = F.elu(x)
        return x


class DynamicChebNet(torch.nn.Module):
    def __init__(self, conv_channels, K=2):
        super(DynamicChebNet, self).__init__()
        self.K = K  # kth order convolution
        self.layers = torch.nn.ModuleList([
            ChebConv(conv_channels[i], conv_channels[i + 1], K)
            for i in range(len(conv_channels) - 1)
        ])

    def forward(self, x, edge_index):
        # Convert sparse matrices to dense matrices
        if isinstance(edge_index, torch.sparse.Tensor):
            edge_index = edge_index.to_dense()

        for layer in self.layers:
            x = F.dropout(x, p=0.3, training=self.training)
            x = layer(x, edge_index)
            x = F.elu(x)
        return x


class CosineSimilarityLoss(torch.nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, data_sets):
        if len(data_sets) < 2:
            raise ValueError("At least two datasets are required for similarity comparison.")

        processed_data_sets = []

        # Spread and store each dataset
        for data in data_sets:
            data = data.flatten()  # Flatten to a 1D vector
            processed_data_sets.append(data)

        # Calculate the average of all data sets
        avg_data = torch.mean(torch.stack(processed_data_sets), dim=0)

        # Calculate the cosine similarity between each dataset and the mean value
        loss_value = 0.0
        for data in processed_data_sets:
            cos_sim = F.cosine_similarity(data, avg_data, dim=0)
            loss_value += 1 / abs(cos_sim)  # The higher the similarity, the lower the loss

        # Taking the average as the final loss
        loss_value /= len(processed_data_sets)

        return loss_value


def filter_edges(data, global_out_data, percentage, similarity_matrix, w1=0.6, w2=0.4):
    """
    Calculate the edge similarity matrix and filter out the specified percentage of edges
    :param similarity_matrix:
    :param data: Contains the edge_index of each graph and the original node features data.x
    :param global_out_data: List of GAT feature matrices for each graph
    :param percentage: The percentage of edges screened, retaining the highest percentage of similarity edges.
    :param w1: Weighting of raw similarity
    :param w2: GAT Weighting of similarity
    :return: Filtered set of edges updated_edge_index, same format as data.edge_index
    """
    edge_index = data.edge_index
    num_nodes = data.x[0].size(0)
    num_edges = edge_index.size(1)

    # Determine if similarity_matrix needs to be calculated
    if similarity_matrix is None:
        # Initialise the cumulative similarity matrix in sparse format
        accumulated_similarity_matrix = lil_matrix((num_nodes, num_nodes), dtype=float)

        # Calculated for each graph data
        for feature_matrix, original_matrix in zip(global_out_data, data.x):
            with torch.no_grad():
                # Move the feature matrix from GPU to CPU and compute Spearman similarity
                original_matrix_cpu = pd.DataFrame(original_matrix.cpu().numpy())
                gat_matrix_cpu = pd.DataFrame(feature_matrix.cpu().numpy())

            # Calculate the Spearman similarity matrix for the original and GAT
            original_similarity_matrix = original_matrix_cpu.T.corr(method='spearman').abs().values
            np.fill_diagonal(original_similarity_matrix, 0)
            gat_similarity_matrix = gat_matrix_cpu.T.corr(method='spearman').abs().values
            np.fill_diagonal(gat_similarity_matrix, 0)

            # Calculate weighted similarity
            sim_matrix = (original_similarity_matrix * w1 + gat_similarity_matrix * w2)

            # Sort the edges in data.edge_index by similarity and assign the value
            edge_values = []
            for i in range(num_edges):
                node_a = edge_index[0, i].item()
                node_b = edge_index[1, i].item()
                similarity = sim_matrix[node_a, node_b]
                edge_values.append((i, node_a, node_b, similarity))

            # Sort edges according to similarity, assigning values hierarchically by percentage
            edge_values.sort(key=lambda x: x[3], reverse=True)  # Sort by similarity descending
            layered_scores = np.zeros(len(edge_values))
            layer_size = max(1, len(edge_values) // 100)

            for i in range(100):
                start_idx = i * layer_size
                end_idx = min((i + 1) * layer_size, len(edge_values))
                for idx in range(start_idx, end_idx):
                    _, node_a, node_b, _ = edge_values[idx]
                    layered_scores[idx] = 100 - i  # Assigning values in descending order from 100

            # Updating the cumulative similarity matrix
            for idx, (_, node_a, node_b, _) in enumerate(edge_values):
                accumulated_similarity_matrix[node_a, node_b] += layered_scores[idx]
                accumulated_similarity_matrix[node_b, node_a] += layered_scores[idx]

        # Convert the cumulative similarity matrix to an average similarity matrix
        accumulated_similarity_matrix = accumulated_similarity_matrix.tocoo()
        accumulated_similarity_matrix.data /= len(global_out_data)
        similarity_matrix = accumulated_similarity_matrix.tocsr()  # Convert to csr format to support row and column indexing

    # Proportionally filter the edges with the highest similarity
    row, col = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    similarity_matrix_dense = similarity_matrix.toarray()
    edge_similarities = similarity_matrix_dense[row, col]
    sorted_indices = np.argsort(edge_similarities)[::-1]  # Sort from highest to lowest
    num_edges_to_keep = int((1 - percentage) * len(sorted_indices))

    # Remove filtered edges
    edges_to_keep = sorted_indices[:num_edges_to_keep]
    keep_row, keep_col = row[edges_to_keep], col[edges_to_keep]

    # Construct a new set of edges updated_edge_index
    updated_edge_index = torch.tensor(np.array([keep_row, keep_col]), device=data.edge_index.device)

    return updated_edge_index, similarity_matrix


def add_edge(data, edge_pathway, global_out_data, percentage, similarity_matrix, w1=0.4, w2=0.6):
    """
    Based on the similarity calculation results of all nodes, select the top percentage of edges to be added.
    :param edge_pathway:
    :param w2:
    :param w1:
    :param th_con:
    :param data: Graph dataset with original feature matrix and edge set
    :param global_out_data: GAT trained feature matrix
    :param percentage: Proportion of screening edges
    :param similarity_matrix: Pre-calculated Spearman similarity matrix
    :param threshold: Thresholds for filtering similarity averages
    :return: Updated edge set and similarity matrix
    """
    num_nodes = data[0].size(0)  # Get the number of nodes (assuming the same number of nodes for each graph)
    edge_index = edge_pathway  # The original set of edges, assuming sparse format

    # If no similarity matrix is passed in, calculate the
    if similarity_matrix is None:
        accumulated_similarity_matrix = coo_matrix((num_nodes, num_nodes))  # Use sparse matrix form

        for feature_matrix, original_matrix in zip(global_out_data, data):
            # Move feature matrix from GPU to CPU for Spearman similarity calculation and disable gradient calculation
            with torch.no_grad():
                original_matrix_cpu = pd.DataFrame(original_matrix.cpu().numpy())
                gat_matrix_cpu = pd.DataFrame(feature_matrix.cpu().numpy())

            # Compute the Spearman similarity matrix
            original_similarity_matrix = original_matrix_cpu.T.corr(method='spearman').abs().values
            np.fill_diagonal(original_similarity_matrix, 0)
            gat_similarity_matrix = gat_matrix_cpu.T.corr(method='spearman').abs().values
            np.fill_diagonal(gat_similarity_matrix, 0)

            # Calculate weighted similarity
            sim_matrix = (original_similarity_matrix * w1 + gat_similarity_matrix * w2)
            # Ranking and hierarchically assigning values to weighted similarities
            triu_indices = np.triu_indices(num_nodes, k=1)
            triu_values = sim_matrix[triu_indices]
            sorted_indices = np.argsort(triu_values)[::-1]
            num_edges = sorted_indices.size
            layered_scores = np.zeros(num_edges)

            # hierarchical assignment of values
            layer_size = max(1, num_edges // 100)
            for i in range(100):
                start_idx = i * layer_size
                end_idx = min((i + 1) * layer_size, num_edges)
                layered_scores[sorted_indices[start_idx:end_idx]] = 100 - i

            # The results of the stratified scores are converted to a sparse matrix and accumulated
            layered_similarity_matrix = coo_matrix((layered_scores, (triu_indices[0], triu_indices[1])),
                                                   shape=(num_nodes, num_nodes))
            accumulated_similarity_matrix += layered_similarity_matrix + layered_similarity_matrix.T  # 确保对称性

        # Find the averaged hierarchical similarity matrix
        average_similarity_matrix = accumulated_similarity_matrix / len(global_out_data)
        similarity_matrix = average_similarity_matrix  # Store a single average similarity sparse matrix
        # Set the similarity value corresponding to the original edge set (edge_pathway) to 0
        for i in range(edge_index.size(1)):
            node_a = edge_index[0, i].item()
            node_b = edge_index[1, i].item()
            similarity_matrix[node_a, node_b] = 0
            similarity_matrix[node_b, node_a] = 0
    else:
        average_similarity_matrix = similarity_matrix
        # Set the lower triangles to 0 and keep the upper triangles
    # Convert average_similarity_matrix to a dense matrix and set the lower triangular part to 0
    average_similarity_matrix = average_similarity_matrix.toarray()  # Converting to NumPy Arrays
    average_similarity_matrix = np.triu(average_similarity_matrix, k=1)

    # Get the similarity value of the upper triangular part on the non-diagonal line and its index
    triu_indices = np.triu_indices(num_nodes, k=1)
    triu_values = np.array(average_similarity_matrix[triu_indices]).flatten()
    num_triu_values = triu_values.size

    if num_triu_values == 0:
        return edge_index, similarity_matrix

    threshold_index = int(edge_index.shape[1] * percentage / 2)
    threshold_index = min(threshold_index, num_triu_values - 1)
    threshold_value = np.partition(triu_values, num_triu_values - threshold_index)[num_triu_values - threshold_index]

    triu_mask = average_similarity_matrix > threshold_value
    frequent_edges = torch.tensor(np.vstack(triu_mask.nonzero()))

    if frequent_edges.size(1) == 0:
        return edge_index, similarity_matrix

    edge_index = torch.cat([edge_index, frequent_edges], dim=1)
    edge_index = remove_duplicate_and_self_loops(edge_index)
    return edge_index, similarity_matrix


def train_model(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()

    out = []

    for item in data.x:
        item = item.to(device)
        feature_matrix = model(item, data.edge_index.to(device))
        out.append(feature_matrix)

    loss_value = criterion(out)
    loss_value.backward()
    optimizer.step()

    # Clearing the cache
    torch.cuda.empty_cache()

    return loss_value.item(), out


def initialize_weights_to_zeros(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.constant_(m.weight, 0)  # Initialise the weights to 0
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)  # Initialise bias to 0


def model_geo(train_data, threshold_f, threshold_a, th_filter_max, th_filter_min, th_add_max, th_add_min, epoch_num,
              save_path, gene_map, model_name, conv_channels, gene_list):
    # Model variable initialisation
    # Model Instantiation
    global model  # Declare model as a global variable
    if model_name == 'GAT':
        model = DynamicGAT(conv_channels).to(device)
    elif model_name == 'GraphSAGE':
        model = DynamicGraphSAGE(conv_channels).to(device)
    elif model_name == 'ChebNet':
        model = DynamicChebNet(conv_channels).to(device)
    elif model_name == 'GCN':
        model = DynamicGCN(conv_channels).to(device)
    model.apply(initialize_weights_to_zeros)
    # parameterisation
    optimizer = optim.Adam(model.parameters(), lr=3e-2, weight_decay=3e-4)
    criterion = CosineSimilarityLoss().to(device)

    global_out_data = []
    # edge_index_array = train_data.edge_index.to(device)
    loss_result = 0
    # De-weighting of edge_index and removal of self-loops before each training session
    train_data.edge_index = remove_duplicate_and_self_loops(train_data.edge_index)
    # -------------------------------------------------------------------------------------------------------
    # Module 1: Parametric training
    loss_num = []
    for epoch in range(30):
        loss_result, global_out_data = train_model(model, train_data, optimizer, criterion)
        loss_num.append(loss_result)
        print(f'Epoch: {epoch:03d}, Loss: {loss_result:.4f}')
    # Saving model parameters
    torch.save(model.state_dict(), 'model_first_phase.pth')

    # Phase 2: Load model parameters and train with smaller learning rates and new weight decays
    model.load_state_dict(torch.load('model_first_phase.pth'))
    # Modifying the learning rate and weight decay of the optimiser
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=3e-6)

    for epoch in range(30, 80):  # Train another 50 epochs in the second phase
        loss_result, global_out_data = train_model(model, train_data, optimizer, criterion)
        loss_num.append(loss_result)
        print(f'Epoch: {epoch:03d}, Loss: {loss_result:.4f}')
    # Loss data retention
    with open(os.path.join(save_path, 'loss_num.txt'), 'w') as f:
        f.write(str(loss_num))

    # Plotting the loss function
    window_length = min(len(loss_num) // 2, 11)  # Ensure that the window length is no more than half of the data length and is an odd number.
    polyorder = 3  # Or 4.

    smooth_loss = savgol_filter(loss_num, window_length=window_length, polyorder=polyorder)

    plt.figure(figsize=(10, 6))
    plt.plot(smooth_loss, color='blue', linewidth=2, label='Smoothed Loss')
    plt.scatter(range(len(loss_num)), loss_num, color='red', label='Loss Values', zorder=5)

    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.xlim(0, len(loss_num) - 1)
    plt.ylim(min(loss_num), max(loss_num))

    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss.png'))

    # ---------------------------------------------------------------------------------------------
    # Modules 2 and 3 to be encapsulated at the back
    # Module 2: Deletion of the adjacency matrix
    loss_all = []
    edge_num_all = []
    for epoch in range(epoch_num):

        print(f'Starting the {epoch}th deletion edge')
        train_data, loss_result = second_part(threshold_f, th_filter_max, th_filter_min, loss_result, global_out_data,
                                              train_data, model, criterion, save_path, epoch)
        edge_num_all.append(train_data.edge_index.size(1))
        if loss_result == -1:
            loss_result = loss_all[-1]
            loss_all.append(loss_result)
        elif loss_result != -1:
            loss_all.append(loss_result)

        print('loss_result after deletion of edges:', loss_result)
        print(f'train_data edge_index size:{train_data.edge_index.size(1)}')
        # Updating the adjacency matrix

        # --------------------------------------------------------------------------------------------------
        # Module 3 Roughing in the additions and also refining them
        # The loss function and feature matrix obtained by calling the updated adjacency matrix
        print(f'Start adding edges at {epoch}.')
        new_edg, loss_result = third_part(threshold_a, th_add_max, th_add_min, loss_result, train_data, model,
                                          criterion, save_path, epoch)

        new_edg_np = new_edg.cpu().numpy().T
        # Convert serial number to corresponding gene name output file
        replaced_new_edg_np = rename_gene(new_edg_np, gene_list, gene_map)
        replaced_new_edg_np.to_csv(os.path.join(save_path, f'gene_new{epoch}.txt'), sep='\t', index=False)

        # The result is output to a file later.
        edge_num_all.append(train_data.edge_index.size(1))
        if loss_result == -1:
            loss_result = loss_all[-1]
            loss_all.append(loss_result)
        elif loss_result != -1:
            loss_all.append(loss_result)
    print('loss_all:', loss_all)
    print('edge_num_all:', edge_num_all)
    x_labels = [f'filter{i // 2 + 1}' if i % 2 == 0 else f'add{i // 2 + 1}' for i in range(len(loss_all))]
    loss_min, loss_max = min(loss_all), max(loss_all)
    loss_y_min, loss_y_max = loss_min * 0.8, loss_max * 1.2

    edge_num_min, edge_num_max = min(edge_num_all), max(edge_num_all)
    edge_num_y_min, edge_num_y_max = edge_num_min * 0.8, edge_num_max * 1.2
    # Creating charts and double y-axes
    # Drawing a Loss Diagram
    fig1, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(x_labels, loss_all, color='b', marker='o', label='Loss')
    ax1.set_xlabel('Epoch', fontsize=16)
    ax1.set_ylabel('Loss', color='b', fontsize=16)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim(loss_y_min, loss_y_max)
    ax1.set_title('Loss per Epoch', fontsize=16)
    ax1.legend(loc='upper right', fontsize=12)
    fig1.tight_layout()
    plt.savefig(os.path.join(save_path, "loss_plot.pdf"), format="pdf")

    # draft Edge Count
    fig2, ax2 = plt.subplots(figsize=(11, 6))
    ax2.plot(x_labels, edge_num_all, color='g', marker='x', label='Edge Count')
    ax2.set_xlabel('Epoch', fontsize=16)
    ax2.set_ylabel('Edge Count', color='g', fontsize=16)
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.set_ylim(edge_num_y_min, edge_num_y_max)
    ax2.set_title('Edge Count per Epoch', fontsize=16)
    ax2.legend(loc='upper right', fontsize=12)
    fig2.tight_layout()
    plt.savefig(os.path.join(save_path, "edge_plot.pdf"), format="pdf")


def second_part(threshold_f, th_max, th_min, loss_result, global_out_data, data, model, criterion, save_path, epoch_f):
    """
       Phase 2: Filtering edges based on similarity and updating the adjacency matrix
       :param th_min:
       :param th_max:
       :param threshold_f: Thresholds for filtering edges
       :param loss_result: Initial loss value
       :param global_out_data: The set of trained feature matrices
       :param data: Chart data sets
       :param model: GAT model
       :param criterion: loss function
       :param save_path: Save Path
       :param epoch_f: The number of epochs for the current training
       :return: Updated dataset and loss
       """
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=8e-7)

    loss_array = []
    edge_index_array = []
    threshold_f_step = 0.02
    mark = True
    loss_array.append(loss_result)
    # The value of loss_array[0] cannot be modified, he is an important marker for the whole function to determine if there is a loss drop or not
    # Marker, threshold, judgement basis for storing threshold update, storage of returned adjacency matrix
    result = [[], [], []]
    result2 = []
    similarity_matrix = None
    while (mark is True) and (threshold_f > th_min) and (threshold_f <= th_max):
        update_edge_index, similarity_matrix = filter_edges(data, global_out_data, threshold_f, similarity_matrix)

        # Updated neighbourhood matrix without weights
        print(f'update_edge_index.size(1):{update_edge_index.size(1)}')

        if update_edge_index.size(1) < data.edge_index.size(1):
            # Used to load the updated feature matrix

            # Save the current model and optimiser state before training
            model_state = model.state_dict()
            optimizer_state = optimizer.state_dict()

            loss_a = []
            for epoch in range(150):
                # The iteration here is to find the minimum loss after the change, compare it with the previous loss, and stop as soon as 99 per cent of the difference between the two losses is close enough to be considered fitted
                loss_num, new_out_data = train_model(model, data, optimizer, criterion)
                loss_a.append(loss_num)
            result[0].append(loss_a)
            result[1].append(round(threshold_f, 2))
            result[2].append(update_edge_index.size(1))

            # New loss function
            last_50_elements = loss_a[-50:]  # Get the last 30 elements of loss_a
            average_loss = sum(last_50_elements) / len(last_50_elements)  # Calculation of average values
            result2.append(
                f'loss_num: {last_50_elements}, threshold_f: {threshold_f}, update_edge_index.size(1): {update_edge_index.size(1)}')
            if len(loss_array) == 1 and average_loss < loss_array[0]:
                # Change the loss_array directly for the first calculation.
                loss_array.append(average_loss)  # Append average to loss_array
                threshold_f += threshold_f_step
                threshold_f = round(threshold_f, 2)
                print(f'threshold_f: {threshold_f}')
                print(f'loss_array: {loss_array}')
                edge_index_array = update_edge_index
                continue
            elif len(loss_array) == 1 and average_loss >= loss_array[0]:
                loss_array.append(average_loss)
                threshold_f -= threshold_f_step
                threshold_f = round(threshold_f, 2)
                print(f'threshold_f: {threshold_f}')
                print(f'loss_array: {loss_array}')
                continue

            # Determine the direction of the threshold update
            if (average_loss < (loss_array[1] * 1.002)) and (average_loss < loss_array[0]):
                threshold_f += threshold_f_step
                threshold_f = round(threshold_f, 2)
                print(f'threshold_f: {threshold_f}')
                edge_index_array = update_edge_index

                # Record only the smallest updated loss_num
                loss_array[1] = average_loss
                print(f'loss_array: {loss_array}')

            elif (average_loss > (loss_array[1] * 1.002)) and (average_loss < loss_array[0]):
                # 已经找到最优的损失，退出循环
                mark = False
                model.load_state_dict(model_state)
                optimizer.load_state_dict(optimizer_state)
                print("Restore the previous parameters as the loss increases")

            elif (average_loss < (loss_array[1] * 1.002)) and (average_loss > loss_array[0]):
                # Finding smaller losses in the direction of decreasing thresholds and returning model parameters prevents them from being modified and wasting more computational time

                threshold_f -= (threshold_f_step / 2)
                threshold_f = round(threshold_f, 2)
                print(f'threshold_f: {threshold_f}')
                loss_array[1] = average_loss
                print(f'loss_array: {loss_array}')
                model.load_state_dict(model_state)
                optimizer.load_state_dict(optimizer_state)

            elif (average_loss > (loss_array[1] * 1.002)) and (average_loss > loss_array[0]):
                # This means that the original result is optimal, no further modifications are needed and the model parameters are returned.

                mark = False
                # If the loss increases, restore the previous model parameters
                model.load_state_dict(model_state)
                optimizer.load_state_dict(optimizer_state)
                print("Restore the previous parameters as the loss increases")

        else:
            # Add threshold if the adjacency matrix has not been updated
            threshold_f += threshold_f_step

    # After updating, if the loss becomes smaller, the number of edges is reduced and then updated
    if len(edge_index_array) > 0:
        edge_index_array = edge_index_array.clone().detach()
        if edge_index_array.size(1) < data.edge_index.size(1):
            data.edge_index = edge_index_array
            data.edge_index = remove_duplicate_and_self_loops(data.edge_index)

    if result[0]:
        # Calculate the minimum value of each array
        min_values = [sum(curve[-50:]) / len(curve[-50:]) for curve in result[0]]

        # Smoothing with Savitzky-Golay Filters
        if len(min_values) >= 5:  # Ensure that the length of min_values is greater than or equal to the length of the window.
            smooth_min_values = savgol_filter(min_values, window_length=5, polyorder=2)
        else:
            smooth_min_values = min_values  # If it is not long enough, use the original minimum value directly

        # Generate the horizontal coordinates, using result[1].
        x_coords = result[1][:len(smooth_min_values)]  # Make sure the lengths match

        # Start drawing
        fig, ax2 = plt.subplots(figsize=(10, 6))

        # Plot result[2] curve below
        ax2.plot(result[1], result[2], label='Result[2]', color='green', linewidth=2)  # Plotting the second curve
        ax2.set_ylabel('Result[2] Values', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        # Adjusting the range of the second y-axis
        ax2.set_ylim(bottom=0)  # Ensure that the lower curve does not overlap with zero

        # Create the first y-axis for the min_values curve.
        ax1 = ax2.twinx()  # Create the first y-axis that shares the x-axis
        ax1.plot(x_coords, smooth_min_values, label='Smoothed Min Values', color='blue', linewidth=2)
        ax1.scatter(x_coords, min_values, color='red', label='Min Values')  # Marks the position of each value

        # Setting the first y-axis
        ax1.set_ylabel('Min Values (result[0])', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Adjusting the y-axis range
        ax1.set_ylim(top=np.max(smooth_min_values) * 1.1)  # Enlarge the range of the upper curve appropriately

        # Add vertical line from top to bottom
        last_x = result[1][-2]  # The last horizontal coordinate
        last_y_result2 = result[2][-2]  # Last value of result[2
        last_y_min = min_values[-2]  # The last minimum value of result[0

        ax1.plot([last_x, last_x], [0, ax1.get_ylim()[1]], color='black', linestyle='--', linewidth=2)

        # Vertical coordinates of the two points labelled with the last coordinate
        ax2.text(last_x, last_y_result2, f'{last_y_result2:.2f}', color='green', ha='right', va='top', fontsize=10)
        ax1.text(last_x, last_y_min, f'{last_y_min:.2f}', color='blue', ha='right', va='bottom', fontsize=10)

        # Mark the vertical coordinates of the yellow line at the point of intersection
        ax2.scatter(last_x, last_y_result2, color='green', zorder=5)  # Mark the yellow dot at the intersection

        # Set all horizontal scales
        ax2.set_xticks(result[1])  # Set all horizontal coordinate values
        ax2.set_xticklabels(result[1], rotation=45)  # Rotate the scale label for better display

        # Add title and legend
        plt.title('Two Curves with Different Y-axes')
        fig.tight_layout()  # Adjust layout to avoid overlapping tags
        plt.savefig(os.path.join(save_path, f'filter_loss{epoch_f}.png'))
        plt.close(fig)

    current_date = date.today()
    filename = f'{current_date}loss_result{epoch_f}.txt'
    with open(os.path.join(save_path, filename), 'w') as file:
        for line in result2:
            file.write(line + '\n')
        for item in result:
            file.write(str(item) + '\n')

    if len(loss_array) == 1:
        return data, -1

    return data, loss_array[1]


def third_part(threshold_a, th_max, th_min, loss_result, data, model, criterion, save_path, epoch_a):
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=8e-7)
    loss_array = []
    edge_index_array = []
    loss_array.append(loss_result)
    threshold_a_step = 0.01
    mark = True
    # The initial value is set to infinity, which makes it easier to determine if a smaller loss value has occurred.
    filter_out = []
    result = [[], [], []]
    result2 = []
    similarity_matrix = None
    for item in data.x:
        item = item.to(device)
        feature_matrix = model(item, data.edge_index.to(device))
        filter_out.append(feature_matrix)
    edge_pathway = data.edge_index
    print(f'Before adding sides:{edge_pathway.size(1)}')

    while (mark is True) and (threshold_a < th_max) and (threshold_a > th_min):
        # Statistics to get the edge to be added
        data.edge_index, similarity_matrix = add_edge(data.x, edge_pathway, filter_out, threshold_a, similarity_matrix)

        print(f'After the side is added:{data.edge_index.size(1)}')

        if edge_pathway.size(1) < data.edge_index.size(1):

            # Save the current model's parameter state before training
            model_state = model.state_dict()
            optimizer_state = optimizer.state_dict()

            # The model is trained to get the loss value after fitting
            loss_a = []
            for epoch in range(100):
                loss_num, new_out_data = train_model(model, data, optimizer, criterion)
                loss_a.append(loss_num)
            result[0].append(loss_a)
            result[1].append(round(threshold_a, 2))
            result[2].append(data.edge_index.size(1))
            result2.append(
                f'loss_num: {min(loss_a)}, threshold_f: {threshold_a}, update_edge_index.size(1): {data.edge_index.size(1)}')
            average_loss = sum(loss_a[-50:]) / len(loss_a[-50:])
            if len(loss_array) == 1 and average_loss < loss_array[0]:
                loss_array.append(average_loss)
                threshold_a -= threshold_a_step
                threshold_a = round(threshold_a, 2)
                print(f'threshold_a: {threshold_a}')
                print(f'loss_array: {loss_array}')
                edge_index_array = data.edge_index
                continue
            elif len(loss_array) == 1 and average_loss > loss_array[0]:
                loss_array.append(average_loss)
                threshold_a += threshold_a_step
                threshold_a = round(threshold_a, 2)
                print(f'threshold_a: {threshold_a}')
                print(f'loss_array: {loss_array}')
                continue

            if (average_loss < (loss_array[1] * 1.002)) and (average_loss < loss_array[0]):
                threshold_a += threshold_a_step
                threshold_a = round(threshold_a, 2)
                print(f'threshold_a: {threshold_a}')
                edge_index_array = data.edge_index
                loss_array[1] = average_loss
                print(f'loss_array: {loss_array}')

            elif (average_loss > (loss_array[1] * 1.002)) and (average_loss < loss_array[0]):
                # The optimal loss has been found, exit the loop
                mark = False
                model.load_state_dict(model_state)
                optimizer.load_state_dict(optimizer_state)
                print("Restore the previous parameters as the loss increases")

            elif (average_loss < (loss_array[1] * 1.002)) and (average_loss > loss_array[0]):

                threshold_a -= threshold_a_step
                print(f'threshold_a: {threshold_a}')
                loss_array[1] = average_loss
                print(f'loss_array: {loss_array}')
                model.load_state_dict(model_state)
                optimizer.load_state_dict(optimizer_state)

            elif (average_loss > (loss_array[1] * 1.002)) and (average_loss > loss_array[0]):
                # This means that the original result is optimal, no further modifications are needed and the model parameters are returned.
                mark = False
                # If the loss increases, restore the previous model parameters
                model.load_state_dict(model_state)
                optimizer.load_state_dict(optimizer_state)
                print("Restore the previous parameters as the loss increases")

        else:
            threshold_a -= threshold_a_step

    if len(edge_index_array) > 0:
        edge_index_array = edge_index_array.clone().detach()
        if edge_index_array.size(1) < data.edge_index.size(1):
            data.edge_index = edge_index_array
            data.edge_index = remove_duplicate_and_self_loops(data.edge_index)

    if result[0]:
        # Calculate the minimum value of each array
        min_values = [sum(curve[-50:]) / len(curve[-50:]) for curve in result[0]]

        # Smoothing with Savitzky-Golay Filters
        if len(min_values) >= 5:  # Ensure that the length of min_values is greater than or equal to the length of the window.
            smooth_min_values = savgol_filter(min_values, window_length=5, polyorder=2)
        else:
            smooth_min_values = min_values  # If it is not long enough, use the original minimum value directly

        # Generate the horizontal coordinates, using result[1].
        x_coords = result[1][:len(smooth_min_values)]  # Make sure the lengths match

        # Start drawing
        fig, ax2 = plt.subplots(figsize=(10, 6))

        # Plot result[2] curve below
        ax2.plot(result[1], result[2], label='Result[2]', color='green', linewidth=2)  # Plotting the second curve
        ax2.set_ylabel('Result[2] Values', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        # Adjusting the range of the second y-axis
        ax2.set_ylim(bottom=0)  # Ensure that the lower curve does not overlap with zero
        # Create the first y-axis for the min_values curve.
        ax1 = ax2.twinx()  # Create the first y-axis that shares the x-axis
        ax1.plot(x_coords, smooth_min_values, label='Smoothed Min Values', color='blue', linewidth=2)
        ax1.scatter(x_coords, min_values, color='red', label='Min Values')  # Marks the position of each value

        # Setting the first y-axis
        ax1.set_ylabel('Min Values (result[0])', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Adjusting the y-axis range
        ax1.set_ylim(top=np.max(smooth_min_values) * 1.1)  # Enlarge the range of the upper curve appropriately

        # Add vertical line from top to bottom
        last_x = result[1][-2]  # The last horizontal coordinate
        last_y_result2 = result[2][-2]  # Last value of result[2]
        last_y_min = min_values[-2]  # The last minimum value of result[0]

        ax1.plot([last_x, last_x], [0, ax1.get_ylim()[1]], color='black', linestyle='--', linewidth=2)

        # Vertical coordinates of the two points labelled with the last coordinate
        ax2.text(last_x, last_y_result2, f'{last_y_result2:.2f}', color='green', ha='right', va='top', fontsize=10)
        ax1.text(last_x, last_y_min, f'{last_y_min:.2f}', color='blue', ha='right', va='bottom', fontsize=10)

        # Mark the vertical coordinates of the yellow line at the point of intersection
        ax2.scatter(last_x, last_y_result2, color='green', zorder=5)  # Mark the yellow dot at the intersection

        # Set all horizontal scales
        ax2.set_xticks(result[1])  # Set all horizontal coordinate values
        ax2.set_xticklabels(result[1], rotation=45)  # Rotate the scale label for better display

        # Add title and legend
        plt.title('Two Curves with Different Y-axes')
        fig.tight_layout()  # Adjust layout to avoid overlapping tags
        plt.savefig(os.path.join(save_path, f'add_loss{epoch_a}.png'))
        plt.close(fig)
    current_date = date.today()
    filename = f'{current_date}third_result{epoch_a}.txt'
    with open(os.path.join(save_path, filename), 'w') as file:
        for line in result2:
            file.write(line + '\n')
        for item in result:
            file.write(str(item) + '\n')

    if len(loss_array) == 1:
        return data.edge_index, -1
    return data.edge_index, loss_array[1]


def rename_gene(new_edg_np, gene_list, gene_map):
    new_edg_np = np.vectorize(str)(new_edg_np)

    replaced_rows = []
    for row in new_edg_np:
        replaced_row = [gene_map.get(item, -1) for item in row]
        replaced_rows.append(replaced_row)

    replaced_new_edg_np = np.array(replaced_rows, dtype=object)

    # Read ID -> Gene name mapping from gene_list
    gene_df = pd.read_csv(gene_list)
    id_to_gene = dict(zip(gene_df.iloc[:, 0].astype(str), gene_df.iloc[:, 1].astype(str)))

    final_result = []
    for row in replaced_new_edg_np:
        new_row = [id_to_gene.get(str(item), None) for item in row]
        if None not in new_row:
            final_result.append(new_row)

    result_df = pd.DataFrame(final_result, columns=['GeneA', 'GeneB'])
    return result_df


def remove_duplicate_and_self_loops(edge_index):
    """
    Violent method: line by line, sorting, de-duplication, self-loop culling, suitable for debugging.
    """
    edge_np = edge_index.cpu().numpy().T
    edge_sorted = np.sort(edge_np, axis=1)
    edge_sorted = edge_sorted[np.lexsort((edge_sorted[:, 1], edge_sorted[:, 0]))]
    unique_edges = []
    prev = None
    for row in edge_sorted:
        u, v = row
        if u == v:
            continue
        if prev is not None and (u == prev[0] and v == prev[1]):  # 重复
            continue
        unique_edges.append([u, v])
        prev = (u, v)

    if not unique_edges:
        return torch.empty((2, 0), dtype=edge_index.dtype, device=edge_index.device)

    edge_array = np.array(unique_edges).T  # shape [2, N]
    return torch.tensor(edge_array, dtype=edge_index.dtype, device=edge_index.device)