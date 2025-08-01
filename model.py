import argparse
import os
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data
from utils.model.class_tool import GAEModel, bce_loss, train_gae, predict_labels, GATModel, GCNModel, mse_loss
from utils.model.edge_tool import model_geo


def class_train(adj_matrix_path, feature_matrix_path, num_classes,
                save_path, epochs, lr, model_class, loss_fn):
    try:
        if not os.path.isabs(save_path):
            save_path = os.path.abspath(save_path)
        if not os.path.isabs(adj_matrix_path):
            adj_matrix_path = os.path.abspath(adj_matrix_path)
        if not os.path.isabs(feature_matrix_path):
            feature_matrix_path = os.path.abspath(feature_matrix_path)

        adj_matrix = pd.read_csv(adj_matrix_path, header=0, sep=r'\s+').iloc[:, :2].values
        edge_index = torch.tensor(adj_matrix[:, :2], dtype=torch.long).t().contiguous()

        feature_matrix = pd.read_csv(feature_matrix_path, header=0, index_col=0)
        features = feature_matrix.iloc[:, :-1].values
        print("Feature matrix shape during training:", features.shape)
        labels = feature_matrix.iloc[:, -1].values
        assert len(set(labels)) == num_classes, "Number of label classes does not match input"

        x = torch.tensor(features, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index)

        # Train the model
        train_gae(data, save_path, epochs, lr, num_classes, model_class, loss_fn)

    except Exception as e:
        print(f"An error occurred: {e}")


def class_predict(adj_matrix_path, feature_matrix_path, model_path, num_classes, output_path, model=GAEModel):
    try:
        if not os.path.isabs(output_path):
            output_path = os.path.abspath(output_path)
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(model_path)
        if not os.path.isabs(feature_matrix_path):
            feature_matrix_path = os.path.abspath(feature_matrix_path)
        if not os.path.isabs(adj_matrix_path):
            adj_matrix_path = os.path.abspath(adj_matrix_path)

        adj_matrix = pd.read_csv(adj_matrix_path, header=0, sep=r'\s+').iloc[:, :2].values
        edge_index = torch.tensor(adj_matrix[:, :2], dtype=torch.long).t().contiguous()

        feature_matrix = pd.read_csv(feature_matrix_path, header=0, index_col=0)
        features = feature_matrix.values

        x = torch.tensor(features, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index)

        predict_labels(data, model, model_path, num_classes, output_path)

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except pd.errors.EmptyDataError as e:
        print(f"Data is empty: {e}")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")


def edge_pre(data_path, feature_dir, gene_map_path, model_name, conv_channels, th_filter, th_add, th_filter_max,
             th_filter_min, th_add_max, th_add_min, epoch, gene_list):
    edge_index = pd.read_csv(data_path, sep='\t', header=0)
    edge_index = edge_index.iloc[:, :2]
    edge_index = edge_index.apply(pd.to_numeric, errors='coerce')
    edge_index = torch.tensor(edge_index.values, dtype=torch.long).t().contiguous()

    x = []
    in_channels = 0
    for root, _, files in os.walk(feature_dir):
        for file in files:
            df = pd.read_csv(os.path.join(root, file), sep='\t', header=0, index_col=0)
            df = df.replace([None, ''], np.nan)
            df.fillna(0, inplace=True)
            df = torch.tensor(df.values, dtype=torch.float)
            in_channels = df.size(1)
            x.append(df)
    data = Data(x=x, edge_index=edge_index)
    save_path = os.path.join(os.path.dirname(data_path), f'result_data_{model_name}_{th_filter}_{th_add}')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    gene_map = {}
    with open(gene_map_path, 'r') as f:
        for line in f:
            value, key = line.strip().split(':')
            value = value.strip()
            key = key.strip()
            gene_map[key] = value
    conv_channels.insert(0, in_channels)
    model_geo(data, th_filter, th_add, th_filter_max, th_filter_min, th_add_max, th_add_min,
              epoch, save_path, gene_map, model_name, conv_channels,  gene_list)


def main():
    parser = argparse.ArgumentParser(description="Graph Autoencoder Model Training and Prediction")
    subparsers = parser.add_subparsers(dest='command')

    # Subparser for training
    train_parser = subparsers.add_parser('train', help="Train the model")
    train_parser.add_argument('--adj_matrix_path', required=True, help="Path to the adjacency matrix")
    train_parser.add_argument('--feature_matrix_path', required=True, help="Path to the feature matrix")
    train_parser.add_argument('--num_classes', type=int, required=True, help="Number of label classes")
    train_parser.add_argument('--save_path', required=True, help="Directory to save the trained model")
    train_parser.add_argument('--epochs', type=int, default=200, help="Number of epochs for training")
    train_parser.add_argument('--lr', type=float, default=0.01, help="Learning rate for training")
    train_parser.add_argument('--model_class', default=GAEModel, help="Model class for training")
    train_parser.add_argument('--loss_fn', default=bce_loss, help="Loss function for training")

    # Subparser for prediction
    predict_parser = subparsers.add_parser('predict', help="Make predictions with the trained model")
    predict_parser.add_argument('--adj_matrix_path', required=True, help="Path to the adjacency matrix")
    predict_parser.add_argument('--feature_matrix_path', required=True, help="Path to the feature matrix")
    predict_parser.add_argument('--model_path', required=True, help="Path to the trained model")
    predict_parser.add_argument('--num_classes', type=int, required=True, help="Number of label classes")
    predict_parser.add_argument('--output_path', required=True, help="Directory to save the predictions")
    predict_parser.add_argument('--model', default=GAEModel, help="Model class for prediction")

    # Subparser for edge preprocessing
    edge_parser = subparsers.add_parser('edge_pre', help="Preprocess edges")
    edge_parser.add_argument('--data_path', required=True, help="Path to the edge data")
    edge_parser.add_argument('--feature_dir', required=True, help="Directory containing feature files")
    edge_parser.add_argument('--gene_map_path', required=True, help="Path to the gene mapping file")
    edge_parser.add_argument('--model_name', required=True, help="Name of the model")
    edge_parser.add_argument('--conv_channels', nargs='+', type=int, required=True, help="Convolutional channels")
    edge_parser.add_argument('--th_filter', type=float, required=True, help="Threshold for filtering")
    edge_parser.add_argument('--th_add', type=float, required=True, help="Threshold for adding edges")
    edge_parser.add_argument('--th_filter_max', type=float, default=0.5, help="Maximum threshold for filtering")
    edge_parser.add_argument('--th_filter_min', type=float, default=0.01, help="Minimum threshold for filtering")
    edge_parser.add_argument('--th_add_max', type=float, default=0.5, help="Maximum threshold for adding edges")
    edge_parser.add_argument('--th_add_min', type=float, default=0.01, help="Minimum threshold for adding edges")
    edge_parser.add_argument('--epoch', type=int, default=1, help="Number of epochs for edge processing")
    edge_parser.add_argument('--gene_list', default="utils/model/gene_list.csv", help="Path to the gene list file")

    args = parser.parse_args()

    if args.command == 'train':
        if args.model_class == 'GAEModel':
            args.model_class = GAEModel
        elif args.model_class == 'GATModel':
            args.model_class = GATModel
        elif args.model_class == 'GCNModel':
            args.model_class = GCNModel
        else:
            raise ValueError("Invalid model class")
        if args.loss_fn == 'bce_loss':
            args.loss_fn = bce_loss
        elif args.loss_fn == 'mse_loss':
            args.loss_fn = mse_loss
        else:
            raise ValueError("Invalid loss function")
        class_train(
            adj_matrix_path=args.adj_matrix_path,
            feature_matrix_path=args.feature_matrix_path,
            num_classes=args.num_classes,
            save_path=args.save_path,
            epochs=args.epochs,
            lr=args.lr,
            model_class=args.model_class,
            loss_fn=args.loss_fn
        )

    elif args.command == 'predict':
        if args.model == 'GAEModel':
            args.model = GAEModel
        elif args.model == 'GATModel':
            args.model = GATModel
        elif args.model == 'GCNModel':
            args.model = GCNModel
        else:
            raise ValueError("Invalid model class")

        class_predict(
            adj_matrix_path=args.adj_matrix_path,
            feature_matrix_path=args.feature_matrix_path,
            model_path=args.model_path,
            num_classes=args.num_classes,
            output_path=args.output_path,
            model=args.model
        )

    elif args.command == 'edge_pre':
        edge_pre(
            data_path=args.data_path,
            feature_dir=args.feature_dir,
            gene_map_path=args.gene_map_path,
            model_name=args.model_name,
            conv_channels=args.conv_channels,
            th_filter=args.th_filter,
            th_add=args.th_add,
            th_filter_max=args.th_filter_max,
            th_filter_min=args.th_filter_min,
            th_add_max=args.th_add_max,
            th_add_min=args.th_add_min,
            epoch=args.epoch,
            gene_list=args.gene_list
        )


if __name__ == '__main__':
    # main()
    edge_pre("./DATA/model-data/gene.txt",  "./DATA/model-data/expression", "./DATA/model-data/gene_map.txt", "GCN", [8,12,9], 0.48, 0.05, 0.5, 0.01, 0.5, 0.46, 1, "./utils/model/gene_list.csv")
