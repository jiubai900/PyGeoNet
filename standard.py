import argparse
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from utils.standard.tool import is_normal_distribution_ks, pearson_similarity, spearman_similarity, plot_ellipse


def statistic(sim_data_path, expr_data_path, output_path, count_threshold=0.8):
    if not os.path.isabs(sim_data_path):
        sim_data_path = os.path.abspath(sim_data_path)
    if not os.path.isabs(expr_data_path):
        expr_data_path = os.path.abspath(expr_data_path)
    if not os.path.isabs(output_path):
        output_path = os.path.abspath(output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    data = pd.DataFrame()
    num = 0

    # Process similarity data and count gene pairs
    for root, dirs, files in os.walk(sim_data_path):
        for file in files:
            num += 1
            df = pd.read_csv(os.path.join(root, file), sep='\t', header=0)
            df = df.iloc[:, :2].astype(str)
            df['count'] = 1

            if data.empty:
                data = df
            else:
                merged = pd.concat([data, df], ignore_index=True)
                data = merged.groupby(['Gene1', 'Gene2'], as_index=False)['count'].sum()
                data = data.drop_duplicates()  # Remove duplicates if any

    # Filter gene pairs that appear in more than 50% of files
    data = data[data['count'] >= round(num * count_threshold)]

    # Create gene mapping
    arr1 = data['Gene1'].tolist()
    arr2 = data['Gene2'].tolist()
    gene = set(arr1).union(set(arr2))
    gene_list = sorted(gene)
    gene_map = {value: index for index, value in enumerate(gene_list)}

    # Map genes to indices
    data['Gene1'] = data['Gene1'].map(gene_map)
    data['Gene2'] = data['Gene2'].map(gene_map)
    # Save gene pairs
    data.to_csv(os.path.join(output_path, 'gene.txt'), sep='\t', index=False)
    with open(os.path.join(output_path, 'gene_map.txt'), 'w') as file:
        for key, value in gene_map.items():
            file.write(f"{key}: {value}\n")
    # Create expression directory
    output_path = os.path.join(output_path, 'expression')
    os.makedirs(output_path, exist_ok=True)

    # Process expression data
    for root, dirs, files in os.walk(expr_data_path):
        for file in files:
            df = pd.read_csv(os.path.join(root, file), sep='\t', header=0)
            columns = df.columns.tolist()
            df[columns[0]] = df[columns[0]].astype(str)

            # Filter genes present in the similarity data
            df_filtered = df[df[columns[0]].isin(gene)]

            # Identify missing genes and add them with zero values
            missing_genes = gene - set(df_filtered[columns[0]])
            new_rows = pd.DataFrame(
                {columns[0]: list(missing_genes), **{col: 0 for col in columns[1:]}}
            )

            # Concatenate filtered and missing rows, sort by gene name
            df_final = pd.concat([df_filtered, new_rows], ignore_index=True)
            df_final = df_final.sort_values(by=columns[0])

            # Save the processed expression data
            df_final.to_csv(f'{os.path.join(output_path, file)[0:-4]}_{df_final.shape[0]}.txt', sep='\t', index=False)


def pca(data_path, png_path, output_path, n_components=5):
    if not os.path.isabs(data_path):
        data_path = os.path.abspath(data_path)
    if not os.path.isabs(png_path):
        png_path = os.path.abspath(png_path)
    if not os.path.exists(png_path):
        os.makedirs(png_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if os.path.isdir(data_path):
        for root, dirs, files in os.walk(data_path):
            for file in files:
                # Read data
                df = pd.read_csv(os.path.join(root, file), sep='\t', index_col=0)
                df = df.astype(float)
                df.fillna(0, inplace=True)

                # Perform PCA analysis and retain 3 principal components
                pca = PCA(n_components)
                principal_components = pca.fit_transform(df)

                # Calculate the explained variance ratio (proportion of variance explained)
                explained_variance_ratio = pca.explained_variance_ratio_

                # Convert PCA results to DataFrame
                pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i + 1}' for i in range(n_components)])
                pca_df = pca_df.round(2)

                features_for_clustering = pca_df[['PC1', 'PC2']]
                pca_df.to_csv(os.path.join(output_path, file), sep='\t', index=True)

                # Use K-means to cluster the first two principal components
                kmeans = KMeans(n_clusters=3, random_state=0).fit(features_for_clustering)
                pca_df['Cluster'] = kmeans.labels_

                # Create clustering plot
                fig, ax = plt.subplots(figsize=(10, 7))
                colors = ['r', 'g', 'b']
                for cluster in pca_df['Cluster'].unique():
                    cluster_df = pca_df[pca_df['Cluster'] == cluster]
                    ax.scatter(cluster_df['PC1'], cluster_df['PC2'],
                               color=colors[cluster], label=f'Cluster {cluster + 1}', s=100, alpha=0.6)

                    mean = cluster_df[['PC1', 'PC2']].mean().values
                    cov = np.cov(cluster_df[['PC1', 'PC2']].T)

                    plot_ellipse(ax, mean, cov, color=colors[cluster], label=f'Cluster {cluster + 1}')

                # Set plot labels and title
                ax.set_xlabel(f'PC1 (Variance: {explained_variance_ratio[0]:.2f})')
                ax.set_ylabel(f'PC2 (Variance: {explained_variance_ratio[1]:.2f})')
                plt.title('Clustering of PCA Components (PC1 vs PC2)')
                plt.legend()
                plt.grid(True)
                save_file = os.path.join(png_path, f'{file}_clustering.png')
                plt.savefig(save_file)
                plt.close()
    else:
        df = pd.read_csv(data_path, sep='\t', index_col=0)
        df = df.astype(float)
        df.fillna(0, inplace=True)

        # Perform PCA analysis and retain 3 principal components
        pca = PCA(n_components)
        principal_components = pca.fit_transform(df)

        # Calculate the explained variance ratio (proportion of variance explained)
        explained_variance_ratio = pca.explained_variance_ratio_

        # Convert PCA results to DataFrame
        pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i + 1}' for i in range(3)])
        pca_df = pca_df.round(2)

        features_for_clustering = pca_df[['PC1', 'PC2']]
        pca_df.to_csv(data_path, sep='\t', index=True)

        # Use K-means to cluster the first two principal components
        kmeans = KMeans(n_clusters=3, random_state=0).fit(features_for_clustering)
        pca_df['Cluster'] = kmeans.labels_

        # Create clustering plot
        fig, ax = plt.subplots(figsize=(10, 7))
        colors = ['r', 'g', 'b']
        for cluster in pca_df['Cluster'].unique():
            cluster_df = pca_df[pca_df['Cluster'] == cluster]
            ax.scatter(cluster_df['PC1'], cluster_df['PC2'],
                       color=colors[cluster], label=f'Cluster {cluster + 1}', s=100, alpha=0.6)

            mean = cluster_df[['PC1', 'PC2']].mean().values
            cov = np.cov(cluster_df[['PC1', 'PC2']].T)

            plot_ellipse(ax, mean, cov, color=colors[cluster], label=f'Cluster {cluster + 1}')

        # Set plot labels and title
        ax.set_xlabel(f'PC1 (Variance: {explained_variance_ratio[0]:.2f})')
        ax.set_ylabel(f'PC2 (Variance: {explained_variance_ratio[1]:.2f})')
        plt.title('Clustering of PCA Components (PC1 vs PC2)')
        plt.legend()
        plt.grid(True)
        save_file = os.path.join(png_path, f'clustering.png')
        plt.savefig(save_file)
        plt.close()


def similarity(data_path, output_path, alpha=0.05, sim_threshold=0.3, num_threads=1, chunk_size=500):
    if not os.path.isabs(data_path):
        data_path = os.path.abspath(data_path)
    if not os.path.isabs(output_path):
        output_path = os.path.abspath(output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)  # Make sure the save path exists
    if os.path.isdir(data_path):
        for root, dirs, files in os.walk(data_path):
            for file in files:
                df = pd.read_csv(os.path.join(root, file), sep='\t', index_col=0)
                data = df.values[:, 1].tolist()
                print(file)
                if is_normal_distribution_ks(data, alpha=alpha):
                    out_path = f'{output_path}/{file}'
                    pearson_similarity(df, out_path, threshold=sim_threshold, thread=num_threads, chunk_size=chunk_size)
                else:
                    out_path = f'{output_path}/{file}'
                    spearman_similarity(df, out_path, threshold=sim_threshold, thread=num_threads,
                                        chunk_size=chunk_size)
    else:
        df = pd.read_csv(data_path, sep='\t', index_col=0)
        data = df.values[:, 1].tolist()
        file = os.path.basename(data_path)
        if is_normal_distribution_ks(data, alpha=alpha):
            out_path = f'{output_path}/{file}'
            pearson_similarity(df, out_path, threshold=sim_threshold, thread=num_threads, chunk_size=chunk_size)
        else:
            out_path = f'{output_path}/{file}'
            spearman_similarity(df, out_path, threshold=sim_threshold, thread=num_threads, chunk_size=chunk_size)


# Main function
def main():
    # Define argument parser
    parser = argparse.ArgumentParser(description="Process GSE data with statistical, PCA, and similarity analysis.")

    # Add arguments
    parser.add_argument("method", choices=["statistic", "pca", "similarity"],
                        help="Choose the operation mode: statistic, pca, or similarity.")
    parser.add_argument("--sim_data_path", type=str, default=None,
                        help="Path to the similarity data directory (required for 'statistic').")
    parser.add_argument("--expr_data_path", type=str, default=None,
                        help="Path to the expression data directory (required for 'statistic').")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to the data file or directory (required for 'pca' and 'similarity').")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to the output directory.")
    parser.add_argument("--png_path", type=str, default=None,
                        help="Path to save PCA clustering plots (required for 'pca').")
    parser.add_argument("--count_threshold", type=float, default=0.8,
                        help="Threshold for counting gene pairs in 'statistic'. Default is 0.8.")
    parser.add_argument("--n_components", type=float, default=5,
                        help="Number of PCA components. Default is 3.")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level for KS-test in 'similarity'. Default is 0.05.")
    parser.add_argument("--sim_threshold", type=float, default=0.3,
                        help="Similarity threshold for Pearson/Spearman. Default is 0.3.")
    parser.add_argument("--num_threads", type=int, default=1,
                        help="Number of threads for similarity calculation. Default is 1.")
    parser.add_argument("--chunk_size", type=int, default=500,
                        help="Chunk size for similarity calculation. Default is 500.")

    # Parse arguments
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    # Perform operations based on mode
    if args.method == "statistic":
        if not args.sim_data_path or not args.expr_data_path:
            raise ValueError("Both --sim_data_path and --expr_data_path are required for 'statistic' mode.")
        statistic(args.sim_data_path, args.expr_data_path, args.output_path, count_threshold=args.count_threshold)
    elif args.method == "pca":
        if not args.data_path or not args.png_path:
            raise ValueError("Both --data_path and --png_path are required for 'pca' mode.")
        pca(args.data_path, args.png_path, args.output_path, n_components=args.n_components)
    elif args.method == "similarity":
        if not args.data_path:
            raise ValueError("--data_path is required for 'similarity' mode.")
        similarity(args.data_path, args.output_path, alpha=args.alpha, sim_threshold=args.sim_threshold,
                   num_threads=args.num_threads, chunk_size=args.chunk_size)
    else:
        raise ValueError(f"Unsupported mode: {args.method}")

    print(f"Operation '{args.method}' completed successfully!")


if __name__ == "__main__":
    main()