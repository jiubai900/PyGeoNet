import os
import numpy as np
import pandas as pd
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm
import numba
from scipy.stats import chi2
from matplotlib.patches import Ellipse


def spearman_similarity(data, save_path, threshold, thread, chunk_size):
    try:  # Exception handling
        # Check if data is empty or contains NaN values
        if data.empty or data.isna().any().any():
            data = data.replace([None, ''], np.nan)
            data.fillna(0, inplace=True)

        # Validate file path existence and permissions
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir) or not os.access(save_dir, os.W_OK):
            raise IOError(f"Cannot write to path: {save_path}")

        # Data type check before conversion to avoid unnecessary type conversions
        if not data.dtypes.apply(lambda col: np.issubdtype(col, np.number)).all():
            data = data.astype(float)

        # Calculate Spearman correlation matrix
        spearman_corr_matrix = data.T.corr(method='spearman')
        np.fill_diagonal(spearman_corr_matrix.values, 0)  # Set diagonal elements to 0

        # Calculate Spearman correlation matrix
        similarity_array = np.round(spearman_corr_matrix.values, decimals=2)
        genes = np.array(data.index)

        print("Starting parallel filtering of Spearman similarity matrix...")
        results = parallel_filter_similarity(similarity_array, genes, threshold, thread, chunk_size)

        # Output results
        spearman_results_df = pd.DataFrame(results, columns=['Gene1', 'Gene2', 'Similarity'])
        spearman_results_df.to_csv(save_path, sep='\t', index=False)

    except Exception as e:
        print(f"Error occurred during processing: {e}")


def pearson_similarity(data, save_path, threshold, thread, chunk_size):
    try:  # Exception handling
        # Check if data is empty or contains NaN values
        if data.empty or data.isna().any().any():
            data = data.replace([None, ''], np.nan)
            data.fillna(0, inplace=True)

        # Validate file path existence and permissions
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir) or not os.access(save_dir, os.W_OK):
            raise IOError(f"Cannot write to path: {save_path}")

        # Data type check before conversion to avoid unnecessary type conversions
        if not data.dtypes.apply(lambda col: np.issubdtype(col, np.number)).all():
            data = data.astype(float)

        pearson_corr_matrix = data.T.corr(method='pearson')

        pearson_corr_matrix_abs = pearson_corr_matrix.abs()
        similarity_array = np.round(pearson_corr_matrix_abs.values, decimals=2)
        genes = np.array(data.index)

        print("Starting parallel filtering of pearson similarity matrix...")
        results = parallel_filter_similarity(similarity_array, genes, threshold, thread, chunk_size)
        pearson_results_df = pd.DataFrame(results, columns=['Gene1', 'Gene2', 'Similarity'])
        pearson_results_df.to_csv(save_path, sep='\t', index=False)
    except Exception as e:
        print(f"Error occurred during processing: {e}")


def is_normal_distribution_ks(data, alpha=0.30):
    data_standardized = (data - np.mean(data)) / np.std(data)
    stat, p_value = stats.kstest(data_standardized, 'norm')
    return p_value > alpha


def plot_ellipse(ax, mean, cov, color, label, alpha=0.3):
    chi2_val = np.sqrt(chi2.ppf(0.95, 2))  # 95% confidence interval corresponding chi-square value
    # Calculate the major axis length and angle of the ellipse
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    eigvals = chi2_val * np.sqrt(eigvals)  # Adjust eigenvalues to reflect 95% confidence interval

    # Calculate the width and height of the ellipse
    width, height = 2.0 * eigvals
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

    # Create the ellipse and add it to the plot
    ellipse = Ellipse(mean, width, height, angle=angle, color=color, fill=True, alpha=alpha)
    ax.add_patch(ellipse)


@numba.jit(nopython=True)
def filter_chunk(start, end, corr_matrix, threshold, n):
    """
    Filter pairs of indices from a correlation matrix that exceed a given threshold.

    Parameters:
    - start: Start index for the chunk
    - end: End index for the chunk
    - corr_matrix: Correlation matrix
    - threshold: Threshold value for filtering
    - n: Number of elements (size of the matrix)

    Returns:
    - chunk_results: List of tuples containing the indices and the corresponding correlation value
    """
    chunk_results = []
    for i in range(start, end):
        for j in range(i + 1, n):
            if corr_matrix[i, j] > threshold:
                chunk_results.append((i, j, corr_matrix[i, j]))
    return chunk_results


def parallel_filter_similarity(corr_matrix, genes, threshold=0.6, thread=3, chunk_size=500):
    """
    Parallelize the filtering of similar gene pairs based on a correlation matrix.

    Parameters:
    - corr_matrix: Correlation matrix
    - genes: List of gene names
    - threshold: Threshold value for filtering (default: 0.6)
    - thread: Number of threads to use (default: 3)
    - chunk_size: Size of each chunk to process (default: 500)

    Returns:
    - results: List of tuples containing the gene names and the corresponding correlation value
    """
    n = len(genes)
    results = []
    with ProcessPoolExecutor(max_workers=thread) as executor:
        futures = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            futures.append(executor.submit(filter_chunk, start, end, corr_matrix, threshold, n))

        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                result = future.result()
                results.extend(result)
            except Exception as e:
                print(f"Error processing chunk: {e}")
    executor.shutdown()

    # Convert indices back to gene names
    results = [(genes[i], genes[j], similarity) for (i, j, similarity) in results]
    return results
