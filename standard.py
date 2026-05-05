import argparse
import os
import threading
from datetime import datetime
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from utils.standard.tool import is_normal_distribution_ks, pearson_similarity, spearman_similarity, plot_ellipse
from utils.log_tool import (
    create_task_logger,
    log_section,
    log_args,
    log_path_status,
    log_exception
)


def standard_log(message):
    """Print standard module progress messages with timestamp and thread name."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    thread_name = threading.current_thread().name
    print(f'[standard][{timestamp}][{thread_name}] {message}', flush=True)


def dataframe_shape(df):
    """Return a safe rows x columns string for debug printing."""
    try:
        return f'{df.shape[0]} rows x {df.shape[1]} columns'
    except Exception:
        return 'unknown shape'


def statistic(sim_data_path, expr_data_path, output_path, count_threshold=0.8,
              log_path=None, task_id=None):
    """
    Statistically screen stable gene pairs from multiple similarity networks
    and generate filtered expression matrices based on the retained gene set.
    """
    logger, log_file = create_task_logger(
        module_name="standard_statistic",
        log_path=log_path,
        task_id=task_id
    )

    total_start_time = time.time()

    log_section(logger, "STATISTIC SCREENING STARTED")
    log_args(
        logger,
        sim_data_path=sim_data_path,
        expr_data_path=expr_data_path,
        output_path=output_path,
        count_threshold=count_threshold,
        log_path=log_path,
        task_id=task_id
    )

    try:
        if not os.path.isabs(sim_data_path):
            sim_data_path = os.path.abspath(sim_data_path)
        if not os.path.isabs(expr_data_path):
            expr_data_path = os.path.abspath(expr_data_path)
        if not os.path.isabs(output_path):
            output_path = os.path.abspath(output_path)

        log_path_status(logger, "Resolved sim_data_path", sim_data_path)
        log_path_status(logger, "Resolved expr_data_path", expr_data_path)
        log_path_status(logger, "Resolved output_path", output_path)

        if not os.path.exists(sim_data_path):
            logger.error(f"Similarity data path does not exist: {sim_data_path}")
            log_section(logger, "STATISTIC SCREENING FAILED")
            logger.info(f"Log file saved at: {log_file}")
            return

        if not os.path.exists(expr_data_path):
            logger.error(f"Expression data path does not exist: {expr_data_path}")
            log_section(logger, "STATISTIC SCREENING FAILED")
            logger.info(f"Log file saved at: {log_file}")
            return

        if not os.path.exists(output_path):
            os.makedirs(output_path)
            logger.info(f"Created statistic output directory: {output_path}")

        try:
            count_threshold = float(count_threshold)
        except Exception as exc:
            log_exception(logger, "Invalid count_threshold parameter.", exc)
            return

        logger.info(f"Final count_threshold: {count_threshold}")

        data = pd.DataFrame()
        num = 0

        log_section(logger, "READING SIMILARITY FILES")

        sim_file_list = []
        for root, dirs, files in os.walk(sim_data_path):
            for file in files:
                sim_file_list.append((root, file))

        logger.info(f"Found {len(sim_file_list)} similarity file(s).")

        if not sim_file_list:
            logger.warning("No similarity file found. Statistic screening stopped.")
            log_section(logger, "STATISTIC SCREENING FINISHED WITH EMPTY INPUT")
            logger.info(f"Log file saved at: {log_file}")
            return

        for index, (root, file) in enumerate(sim_file_list, start=1):
            input_file = os.path.join(root, file)

            logger.info("-" * 100)
            logger.info(f"[{index}/{len(sim_file_list)}] Reading similarity file: {input_file}")

            try:
                num += 1

                df = pd.read_csv(input_file, sep='\t', header=0)
                logger.info(f"[{file}] Raw similarity data shape: {df.shape[0]} rows x {df.shape[1]} columns")
                logger.info(f"[{file}] Raw columns preview: {list(df.columns[:10])}")

                if df.shape[1] < 2:
                    logger.warning(f"[{file}] Fewer than 2 columns. Skip this file.")
                    num -= 1
                    continue

                df = df.iloc[:, :2].astype(str)
                df['count'] = 1

                logger.info(f"[{file}] Candidate gene pairs used for counting: {df.shape[0]}")

                if data.empty:
                    data = df
                    logger.info(f"[{file}] Initialized merged candidate relationship table.")
                else:
                    before_concat_count = data.shape[0]
                    merged = pd.concat([data, df], ignore_index=True)
                    logger.info(
                        f"[{file}] Concatenated candidate relationships: "
                        f"before={before_concat_count}, current_file={df.shape[0]}, merged={merged.shape[0]}"
                    )

                    data = merged.groupby(['Gene1', 'Gene2'], as_index=False)['count'].sum()
                    data = data.drop_duplicates()

                    logger.info(f"[{file}] Unique candidate relationships after grouping: {data.shape[0]}")

            except Exception as exc:
                num -= 1
                log_exception(logger, f"[{file}] Failed to process similarity file.", exc)
                continue

        logger.info("-" * 100)
        logger.info(f"Valid similarity file count: {num}")

        if num == 0:
            logger.warning("No valid similarity file was processed. Statistic screening stopped.")
            log_section(logger, "STATISTIC SCREENING FINISHED WITH NO VALID SIMILARITY FILE")
            logger.info(f"Log file saved at: {log_file}")
            return

        if data.empty:
            logger.warning("Merged candidate relationship table is empty. Statistic screening stopped.")
            log_section(logger, "STATISTIC SCREENING FINISHED WITH EMPTY RELATIONSHIP TABLE")
            logger.info(f"Log file saved at: {log_file}")
            return

        log_section(logger, "FILTERING STABLE CANDIDATE RELATIONSHIPS")

        before_filter_count = data.shape[0]
        required_count = round(num * count_threshold)

        logger.info(f"Candidate relationships before filtering: {before_filter_count}")
        logger.info("Filter rule: count >= round(valid_file_count * count_threshold)")
        logger.info(f"Required count: round({num} * {count_threshold}) = {required_count}")

        data = data[data['count'] >= required_count]

        after_filter_count = data.shape[0]

        logger.info(f"Candidate relationships after filtering: {after_filter_count}")
        logger.info(f"Removed candidate relationships: {before_filter_count - after_filter_count}")

        if data.empty:
            logger.warning("No candidate relationship remained after statistical filtering.")
            log_section(logger, "STATISTIC SCREENING FINISHED WITH EMPTY FILTERED RESULT")
            logger.info(f"Log file saved at: {log_file}")
            return

        log_section(logger, "CREATING GENE MAPPING")

        arr1 = data['Gene1'].tolist()
        arr2 = data['Gene2'].tolist()
        gene = set(arr1).union(set(arr2))
        gene_list = sorted(gene)
        gene_map = {value: index for index, value in enumerate(gene_list)}

        logger.info(f"Retained candidate relationship count: {data.shape[0]}")
        logger.info(f"Unique gene count in retained relationships: {len(gene_list)}")
        logger.info(f"Gene preview: {gene_list[:20]}")

        data['Gene1'] = data['Gene1'].map(gene_map)
        data['Gene2'] = data['Gene2'].map(gene_map)

        gene_file = os.path.join(output_path, 'gene.txt')
        gene_map_file = os.path.join(output_path, 'gene_map.txt')

        data.to_csv(gene_file, sep='\t', index=False)
        logger.info(f"Filtered candidate relationships saved to: {gene_file}")
        logger.info(f"gene.txt shape: {data.shape[0]} rows x {data.shape[1]} columns")

        with open(gene_map_file, 'w', encoding='utf-8') as file:
            for key, value in gene_map.items():
                file.write(f"{key}: {value}\n")

        logger.info(f"Gene mapping saved to: {gene_map_file}")
        logger.info(f"Gene mapping count: {len(gene_map)}")

        expression_output_path = os.path.join(output_path, 'expression')
        os.makedirs(expression_output_path, exist_ok=True)

        logger.info(f"Expression output directory: {expression_output_path}")

        log_section(logger, "PROCESSING EXPRESSION FILES")

        expr_file_list = []
        for root, dirs, files in os.walk(expr_data_path):
            for file in files:
                expr_file_list.append((root, file))

        logger.info(f"Found {len(expr_file_list)} expression file(s).")

        if not expr_file_list:
            logger.warning(
                "No expression file found. gene.txt and gene_map.txt have been generated, "
                "but expression output is empty."
            )
            log_section(logger, "STATISTIC SCREENING FINISHED WITHOUT EXPRESSION FILES")
            logger.info(f"Log file saved at: {log_file}")
            return

        expr_success_count = 0
        expr_failed_count = 0

        for index, (root, file) in enumerate(expr_file_list, start=1):
            input_file = os.path.join(root, file)

            logger.info("-" * 100)
            logger.info(f"[{index}/{len(expr_file_list)}] Reading expression file: {input_file}")

            try:
                df = pd.read_csv(input_file, sep='\t', header=0)

                logger.info(f"[{file}] Raw expression data shape: {df.shape[0]} rows x {df.shape[1]} columns")
                logger.info(f"[{file}] Raw columns preview: {list(df.columns[:10])}")

                if df.shape[1] < 2:
                    logger.warning(f"[{file}] Expression file has fewer than 2 columns. Skip this file.")
                    expr_failed_count += 1
                    continue

                columns = df.columns.tolist()
                gene_column = columns[0]

                logger.info(f"[{file}] Gene column used for filtering: {gene_column}")

                df[gene_column] = df[gene_column].astype(str)

                df_filtered = df[df[gene_column].isin(gene)]

                retained_gene_count = df_filtered.shape[0]
                missing_genes = gene - set(df_filtered[gene_column])
                missing_gene_count = len(missing_genes)

                logger.info(f"[{file}] Genes retained from expression file: {retained_gene_count}")
                logger.info(f"[{file}] Missing genes to be filled with zero: {missing_gene_count}")

                new_rows = pd.DataFrame(
                    {gene_column: list(missing_genes), **{col: 0 for col in columns[1:]}}
                )

                df_final = pd.concat([df_filtered, new_rows], ignore_index=True)
                df_final = df_final.sort_values(by=gene_column)

                output_file = f'{os.path.join(expression_output_path, file)[0:-4]}_{df_final.shape[0]}.txt'

                df_final.to_csv(output_file, sep='\t', index=False)

                logger.info(f"[{file}] Final expression data shape: {df_final.shape[0]} rows x {df_final.shape[1]} columns")
                logger.info(f"[{file}] Processed expression file saved to: {output_file}")

                expr_success_count += 1

            except Exception as exc:
                expr_failed_count += 1
                log_exception(logger, f"[{file}] Failed to process expression file.", exc)
                continue

        total_elapsed = time.time() - total_start_time

        log_section(logger, "STATISTIC SCREENING FINISHED")
        logger.info(f"Valid similarity files: {num}")
        logger.info(f"Retained candidate relationships: {after_filter_count}")
        logger.info(f"Retained unique genes: {len(gene_list)}")
        logger.info(f"Expression files processed successfully: {expr_success_count}")
        logger.info(f"Expression files failed: {expr_failed_count}")
        logger.info(f"Total elapsed time: {total_elapsed:.2f} seconds")
        logger.info(f"Log file saved at: {log_file}")

    except Exception as exc:
        log_exception(logger, "Unexpected error occurred in statistic screening.", exc)
        log_section(logger, "STATISTIC SCREENING FAILED")
        logger.info(f"Log file saved at: {log_file}")
        return


def pca(data_path, png_path, output_path, n_components=5):
    standard_log("PCA module started.")
    standard_log(
        f"Input arguments: data_path={data_path}, png_path={png_path}, "
        f"output_path={output_path}, n_components={n_components}"
    )

    try:
        n_components = int(n_components)
    except Exception:
        standard_log(f"Invalid n_components={n_components}. Use default n_components=5.")
        n_components = 5

    if not os.path.isabs(data_path):
        data_path = os.path.abspath(data_path)
    if not os.path.isabs(png_path):
        png_path = os.path.abspath(png_path)
    if not os.path.isabs(output_path):
        output_path = os.path.abspath(output_path)

    standard_log(f"Resolved data_path: {data_path}")
    standard_log(f"Resolved png_path: {png_path}")
    standard_log(f"Resolved output_path: {output_path}")

    if not os.path.exists(png_path):
        os.makedirs(png_path)
        standard_log(f"Created PCA figure directory: {png_path}")

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        standard_log(f"Created PCA output directory: {output_path}")

    if os.path.isdir(data_path):
        standard_log("PCA running in batch-directory mode.")

        file_list = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                file_list.append((root, file))

        standard_log(f"Found {len(file_list)} file(s) for PCA.")

        if not file_list:
            standard_log("No input file found. PCA module stopped.")
            return

        for index, (root, file) in enumerate(file_list, start=1):
            input_file = os.path.join(root, file)
            output_file = os.path.join(output_path, file)
            save_file = os.path.join(png_path, f'{file}_clustering.png')

            standard_log(f"[{index}/{len(file_list)}] Processing PCA file: {input_file}")

            try:
                # Read data
                df = pd.read_csv(input_file, sep='\t', index_col=0)
                standard_log(f"[{file}] Raw data loaded: {dataframe_shape(df)}")

                df = df.astype(float)
                df.fillna(0, inplace=True)
                standard_log(f"[{file}] Data converted to float and missing values filled with 0.")

                available_components = min(n_components, df.shape[0], df.shape[1])
                if available_components < n_components:
                    standard_log(
                        f"[{file}] n_components={n_components} is larger than data limit. "
                        f"Use n_components={available_components} instead."
                    )

                if available_components < 2:
                    standard_log(
                        f"[{file}] PCA skipped because available_components={available_components} < 2."
                    )
                    continue

                # Perform PCA analysis
                pca_model = PCA(n_components=available_components)
                principal_components = pca_model.fit_transform(df)

                # Calculate the explained variance ratio
                explained_variance_ratio = pca_model.explained_variance_ratio_
                variance_text = ', '.join(
                    [f'PC{i + 1}={ratio:.4f}' for i, ratio in enumerate(explained_variance_ratio)]
                )
                standard_log(f"[{file}] PCA finished. Explained variance ratio: {variance_text}")

                # Convert PCA results to DataFrame
                pca_df = pd.DataFrame(
                    data=principal_components,
                    columns=[f'PC{i + 1}' for i in range(available_components)]
                )
                pca_df = pca_df.round(2)

                features_for_clustering = pca_df[['PC1', 'PC2']]
                pca_df.to_csv(output_file, sep='\t', index=True)
                standard_log(f"[{file}] PCA result saved to: {output_file}")

                # Use K-means to cluster the first two principal components
                kmeans = KMeans(n_clusters=3, random_state=0).fit(features_for_clustering)
                pca_df['Cluster'] = kmeans.labels_
                standard_log(f"[{file}] KMeans clustering finished. n_clusters=3")

                # Create clustering plot
                fig, ax = plt.subplots(figsize=(10, 7))
                colors = ['r', 'g', 'b']

                for cluster in pca_df['Cluster'].unique():
                    cluster_df = pca_df[pca_df['Cluster'] == cluster]
                    ax.scatter(
                        cluster_df['PC1'],
                        cluster_df['PC2'],
                        color=colors[cluster],
                        label=f'Cluster {cluster + 1}',
                        s=100,
                        alpha=0.6
                    )

                    mean = cluster_df[['PC1', 'PC2']].mean().values
                    cov = np.cov(cluster_df[['PC1', 'PC2']].T)

                    plot_ellipse(ax, mean, cov, color=colors[cluster], label=f'Cluster {cluster + 1}')

                # Set plot labels and title
                ax.set_xlabel(f'PC1 (Variance: {explained_variance_ratio[0]:.2f})')
                ax.set_ylabel(f'PC2 (Variance: {explained_variance_ratio[1]:.2f})')
                plt.title('Clustering of PCA Components (PC1 vs PC2)')
                plt.legend()
                plt.grid(True)
                plt.savefig(save_file)
                plt.close()

                standard_log(f"[{file}] PCA clustering figure saved to: {save_file}")
                standard_log(f"[{file}] PCA processing completed.")

            except Exception as e:
                standard_log(f"[{file}] PCA processing failed. Error: {e}")
                continue

        standard_log("PCA module finished in batch-directory mode.")

    else:
        standard_log("PCA running in single-file mode.")
        file = os.path.basename(data_path)
        save_file = os.path.join(png_path, 'clustering.png')

        try:
            df = pd.read_csv(data_path, sep='\t', index_col=0)
            standard_log(f"[{file}] Raw data loaded: {dataframe_shape(df)}")

            df = df.astype(float)
            df.fillna(0, inplace=True)
            standard_log(f"[{file}] Data converted to float and missing values filled with 0.")

            available_components = min(n_components, df.shape[0], df.shape[1])
            if available_components < n_components:
                standard_log(
                    f"[{file}] n_components={n_components} is larger than data limit. "
                    f"Use n_components={available_components} instead."
                )

            if available_components < 2:
                standard_log(
                    f"[{file}] PCA skipped because available_components={available_components} < 2."
                )
                return

            # Perform PCA analysis
            pca_model = PCA(n_components=available_components)
            principal_components = pca_model.fit_transform(df)

            # Calculate the explained variance ratio
            explained_variance_ratio = pca_model.explained_variance_ratio_
            variance_text = ', '.join(
                [f'PC{i + 1}={ratio:.4f}' for i, ratio in enumerate(explained_variance_ratio)]
            )
            standard_log(f"[{file}] PCA finished. Explained variance ratio: {variance_text}")

            # Convert PCA results to DataFrame
            pca_df = pd.DataFrame(
                data=principal_components,
                columns=[f'PC{i + 1}' for i in range(available_components)]
            )
            pca_df = pca_df.round(2)

            features_for_clustering = pca_df[['PC1', 'PC2']]
            pca_df.to_csv(data_path, sep='\t', index=True)
            standard_log(f"[{file}] PCA result saved to: {data_path}")

            # Use K-means to cluster the first two principal components
            kmeans = KMeans(n_clusters=3, random_state=0).fit(features_for_clustering)
            pca_df['Cluster'] = kmeans.labels_
            standard_log(f"[{file}] KMeans clustering finished. n_clusters=3")

            # Create clustering plot
            fig, ax = plt.subplots(figsize=(10, 7))
            colors = ['r', 'g', 'b']

            for cluster in pca_df['Cluster'].unique():
                cluster_df = pca_df[pca_df['Cluster'] == cluster]
                ax.scatter(
                    cluster_df['PC1'],
                    cluster_df['PC2'],
                    color=colors[cluster],
                    label=f'Cluster {cluster + 1}',
                    s=100,
                    alpha=0.6
                )

                mean = cluster_df[['PC1', 'PC2']].mean().values
                cov = np.cov(cluster_df[['PC1', 'PC2']].T)

                plot_ellipse(ax, mean, cov, color=colors[cluster], label=f'Cluster {cluster + 1}')

            # Set plot labels and title
            ax.set_xlabel(f'PC1 (Variance: {explained_variance_ratio[0]:.2f})')
            ax.set_ylabel(f'PC2 (Variance: {explained_variance_ratio[1]:.2f})')
            plt.title('Clustering of PCA Components (PC1 vs PC2)')
            plt.legend()
            plt.grid(True)
            plt.savefig(save_file)
            plt.close()

            standard_log(f"[{file}] PCA clustering figure saved to: {save_file}")
            standard_log("PCA module finished in single-file mode.")

        except Exception as e:
            standard_log(f"[{file}] PCA processing failed. Error: {e}")
            raise


def similarity(data_path, output_path, alpha=0.05, sim_threshold=0.3,
               num_threads=1, chunk_size=500, log_path=None, task_id=None):
    """
    Calculate gene similarity network from expression data.

    The function automatically selects Pearson or Spearman similarity according
    to the KS normality test result.
    """
    logger, log_file = create_task_logger(
        module_name="standard_similarity",
        log_path=log_path,
        task_id=task_id
    )

    total_start_time = time.time()

    log_section(logger, "SIMILARITY CALCULATION STARTED")
    log_args(
        logger,
        data_path=data_path,
        output_path=output_path,
        alpha=alpha,
        sim_threshold=sim_threshold,
        num_threads=num_threads,
        chunk_size=chunk_size,
        log_path=log_path,
        task_id=task_id
    )

    try:
        if not os.path.isabs(data_path):
            data_path = os.path.abspath(data_path)
        if not os.path.isabs(output_path):
            output_path = os.path.abspath(output_path)

        log_path_status(logger, "Resolved data_path", data_path)
        log_path_status(logger, "Resolved output_path", output_path)

        if not os.path.exists(data_path):
            logger.error(f"Input data path does not exist: {data_path}")
            log_section(logger, "SIMILARITY CALCULATION FAILED")
            logger.info(f"Log file saved at: {log_file}")
            return

        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
            logger.info(f"Created similarity output directory: {output_path}")

        try:
            alpha = float(alpha)
            sim_threshold = float(sim_threshold)
            num_threads = int(num_threads)
            chunk_size = int(chunk_size)
        except Exception as exc:
            log_exception(logger, "Invalid numeric parameter for similarity calculation.", exc)
            return

        logger.info(f"Final alpha: {alpha}")
        logger.info(f"Final sim_threshold: {sim_threshold}")
        logger.info(f"Final num_threads: {num_threads}")
        logger.info(f"Final chunk_size: {chunk_size}")

        if os.path.isdir(data_path):
            log_section(logger, "BATCH SIMILARITY MODE")

            file_list = []
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    file_list.append((root, file))

            logger.info(f"Found {len(file_list)} input file(s) for similarity calculation.")

            if not file_list:
                logger.warning("No input file found. Similarity calculation stopped.")
                log_section(logger, "SIMILARITY CALCULATION FINISHED WITH EMPTY INPUT")
                logger.info(f"Log file saved at: {log_file}")
                return

            success_count = 0
            failed_count = 0

            for index, (root, file) in enumerate(file_list, start=1):
                file_start_time = time.time()

                input_file = os.path.join(root, file)
                out_path = os.path.join(output_path, file)

                log_section(logger, f"PROCESSING FILE {index}/{len(file_list)}: {file}")
                logger.info(f"Input file: {input_file}")
                logger.info(f"Output file: {out_path}")

                try:
                    df = pd.read_csv(input_file, sep='\t', index_col=0)
                    logger.info(f"[{file}] Data loaded successfully.")
                    logger.info(f"[{file}] Data shape: {df.shape[0]} rows x {df.shape[1]} columns")
                    logger.info(f"[{file}] Data columns preview: {list(df.columns[:10])}")

                    if df.shape[1] < 2:
                        logger.warning(
                            f"[{file}] Data has fewer than 2 columns after index_col=0. "
                            f"Skip this file."
                        )
                        failed_count += 1
                        continue

                    data = df.values[:, 1].tolist()
                    logger.info(f"[{file}] KS-test data length: {len(data)}")
                    logger.info(f"[{file}] KS-test alpha: {alpha}")

                    is_normal = is_normal_distribution_ks(data, alpha=alpha)
                    logger.info(f"[{file}] KS normality test result: {is_normal}")

                    if is_normal:
                        logger.info(f"[{file}] Selected similarity method: Pearson")
                        logger.info(
                            f"[{file}] Start Pearson similarity calculation. "
                            f"threshold={sim_threshold}, thread={num_threads}, chunk_size={chunk_size}"
                        )

                        pearson_similarity(
                            df,
                            out_path,
                            threshold=sim_threshold,
                            thread=num_threads,
                            chunk_size=chunk_size
                        )

                    else:
                        logger.info(f"[{file}] Selected similarity method: Spearman")
                        logger.info(
                            f"[{file}] Start Spearman similarity calculation. "
                            f"threshold={sim_threshold}, thread={num_threads}, chunk_size={chunk_size}"
                        )

                        spearman_similarity(
                            df,
                            out_path,
                            threshold=sim_threshold,
                            thread=num_threads,
                            chunk_size=chunk_size
                        )

                    if os.path.exists(out_path):
                        logger.info(f"[{file}] Similarity result saved to: {out_path}")
                        logger.info(f"[{file}] Output file size: {os.path.getsize(out_path)} bytes")
                    else:
                        logger.warning(
                            f"[{file}] Similarity function finished, but output file was not found: {out_path}"
                        )

                    file_elapsed = time.time() - file_start_time
                    logger.info(f"[{file}] Similarity calculation completed.")
                    logger.info(f"[{file}] Elapsed time: {file_elapsed:.2f} seconds")

                    success_count += 1

                except Exception as exc:
                    failed_count += 1
                    log_exception(logger, f"[{file}] Similarity calculation failed.", exc)
                    continue

            total_elapsed = time.time() - total_start_time

            log_section(logger, "SIMILARITY CALCULATION FINISHED")
            logger.info(f"Total input files: {len(file_list)}")
            logger.info(f"Successful files: {success_count}")
            logger.info(f"Failed files: {failed_count}")
            logger.info(f"Total elapsed time: {total_elapsed:.2f} seconds")
            logger.info(f"Log file saved at: {log_file}")

        else:
            log_section(logger, "SINGLE-FILE SIMILARITY MODE")

            file_start_time = time.time()
            file = os.path.basename(data_path)
            out_path = os.path.join(output_path, file)

            logger.info(f"Input file: {data_path}")
            logger.info(f"Output file: {out_path}")

            try:
                df = pd.read_csv(data_path, sep='\t', index_col=0)
                logger.info(f"[{file}] Data loaded successfully.")
                logger.info(f"[{file}] Data shape: {df.shape[0]} rows x {df.shape[1]} columns")
                logger.info(f"[{file}] Data columns preview: {list(df.columns[:10])}")

                if df.shape[1] < 2:
                    logger.error(
                        f"[{file}] Data has fewer than 2 columns after index_col=0. "
                        f"Similarity calculation stopped."
                    )
                    return

                data = df.values[:, 1].tolist()
                logger.info(f"[{file}] KS-test data length: {len(data)}")
                logger.info(f"[{file}] KS-test alpha: {alpha}")

                is_normal = is_normal_distribution_ks(data, alpha=alpha)
                logger.info(f"[{file}] KS normality test result: {is_normal}")

                if is_normal:
                    logger.info(f"[{file}] Selected similarity method: Pearson")
                    logger.info(
                        f"[{file}] Start Pearson similarity calculation. "
                        f"threshold={sim_threshold}, thread={num_threads}, chunk_size={chunk_size}"
                    )

                    pearson_similarity(
                        df,
                        out_path,
                        threshold=sim_threshold,
                        thread=num_threads,
                        chunk_size=chunk_size
                    )

                else:
                    logger.info(f"[{file}] Selected similarity method: Spearman")
                    logger.info(
                        f"[{file}] Start Spearman similarity calculation. "
                        f"threshold={sim_threshold}, thread={num_threads}, chunk_size={chunk_size}"
                    )

                    spearman_similarity(
                        df,
                        out_path,
                        threshold=sim_threshold,
                        thread=num_threads,
                        chunk_size=chunk_size
                    )

                if os.path.exists(out_path):
                    logger.info(f"[{file}] Similarity result saved to: {out_path}")
                    logger.info(f"[{file}] Output file size: {os.path.getsize(out_path)} bytes")
                else:
                    logger.warning(
                        f"[{file}] Similarity function finished, but output file was not found: {out_path}"
                    )

                file_elapsed = time.time() - file_start_time
                total_elapsed = time.time() - total_start_time

                log_section(logger, "SIMILARITY CALCULATION FINISHED")
                logger.info(f"[{file}] Elapsed time: {file_elapsed:.2f} seconds")
                logger.info(f"Total elapsed time: {total_elapsed:.2f} seconds")
                logger.info(f"Log file saved at: {log_file}")

            except Exception as exc:
                log_exception(logger, f"[{file}] Similarity calculation failed.", exc)
                log_section(logger, "SIMILARITY CALCULATION FAILED")
                logger.info(f"Log file saved at: {log_file}")
                return

    except Exception as exc:
        log_exception(logger, "Unexpected error occurred in similarity calculation.", exc)
        log_section(logger, "SIMILARITY CALCULATION FAILED")
        logger.info(f"Log file saved at: {log_file}")
        return


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
    parser.add_argument("--n_components", type=int, default=5,
                        help="Number of PCA components. Default is 5.")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level for KS-test in 'similarity'. Default is 0.05.")
    parser.add_argument("--sim_threshold", type=float, default=0.3,
                        help="Similarity threshold for Pearson/Spearman. Default is 0.3.")
    parser.add_argument("--num_threads", type=int, default=1,
                        help="Number of threads for similarity calculation. Default is 1.")
    parser.add_argument("--chunk_size", type=int, default=500,
                        help="Chunk size for similarity calculation. Default is 500.")
    parser.add_argument("--log_path", type=str, default=None,
                        help="Path to save log files.")
    parser.add_argument("--task_id", type=str, default=None,
                        help="Optional task id for current run.")

    # Parse arguments
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    # Perform operations based on mode
    if args.method == "statistic":
        if not args.sim_data_path or not args.expr_data_path:
            raise ValueError("Both --sim_data_path and --expr_data_path are required for 'statistic' mode.")
        statistic(
            args.sim_data_path,
            args.expr_data_path,
            args.output_path,
            count_threshold=args.count_threshold,
            log_path=args.log_path,
            task_id=args.task_id
        )
    elif args.method == "pca":
        if not args.data_path or not args.png_path:
            raise ValueError("Both --data_path and --png_path are required for 'pca' mode.")
        pca(args.data_path, args.png_path, args.output_path, n_components=args.n_components)
    elif args.method == "similarity":
        if not args.data_path:
            raise ValueError("--data_path is required for 'similarity' mode.")
        similarity(
            args.data_path,
            args.output_path,
            alpha=args.alpha,
            sim_threshold=args.sim_threshold,
            num_threads=args.num_threads,
            chunk_size=args.chunk_size,
            log_path=args.log_path,
            task_id=args.task_id
        )
    else:
        raise ValueError(f"Unsupported mode: {args.method}")

    print(f"Operation '{args.method}' completed successfully!")


if __name__ == "__main__":
    main()
