import argparse
import os
import warnings
import gseapy as gp
from gseapy.plot import barplot, dotplot
import numpy as np
import pandas as pd
import umap
from scipy.stats import ttest_ind
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from matplotlib.colorbar import ColorbarBase
from scipy.stats import pearsonr

from utils.analyse.tool import read_data, get_point, benjamini_hochberg, get_signpoint, cluster_calculate, ench_read_data


def volcano(file_path, save_path='./', node_num=10):
    if os.path.isabs(save_path):
        save_path = os.path.abspath(save_path)
    if not os.path.isabs(file_path):
        file_path = os.path.abspath(file_path)

    df = pd.read_csv(file_path, sep='\t', low_memory=False)
    df, num_gsm, group_df, group_num, group_arr = read_data(df)

    for i in range(group_num):
        for j in range(i + 1, group_num):

            significant_points, non_significant_points, point, calculate_df = get_point(group_df[i], group_df[j],
                                                                                        num_gsm)

            # Output the salient points
            df_result = significant_points[['log10(p)', 'log2FC']]
            df_result = df_result.reset_index()
            df_result.columns = ['gene', 'p-value', 'Fold Change']
            p_value = df_result['p-value'].values
            adjusted_p_values = benjamini_hochberg(p_value)
            df_result['FDR'] = adjusted_p_values
            file_name = f'{group_arr[i]}_{group_arr[j]}.txt'
            df_result.to_csv(os.path.join(save_path, file_name), sep='\t', index=False)

            # plot
            fig, ax = plt.subplots(figsize=(8, 6))
            plt.axvline(x=0, color=".5")
            plt.axhline(y=-np.log10(0.05), linestyle='--')
            plt.axvline(x=1, linestyle='--')  # green dotted line
            plt.axvline(x=-1, linestyle='--')  # green dotted line
            ax.scatter(calculate_df['log2FC'], calculate_df['log10(p)'], color='grey')  # all points
            ax.scatter(significant_points['log2FC'], significant_points['log10(p)'], color='red', label='Significant '
                                                                                                        'Points')  # remarkable point
            ax.scatter(non_significant_points['log2FC'], non_significant_points['log10(p)'], color='blue')  # non-significant point
            ax.scatter(point['log2FC'], point['log10(p)'], color='green')  # non-significant point
            # 添加轴标签和标题
            ax.set_xlabel('Log2 Fold Change')
            ax.set_ylabel('-log10(p-value)')
            ax.set_title(f'{group_arr[i]}-{group_arr[j]} Volcano Plot')

            # Add Legend
            ax.legend()

            # Optional: set limits for x and y axes
            ax.set_xlim([calculate_df['log2FC'].min() - 1, calculate_df['log2FC'].max() + 1])  # Adjustment of the scope of FC
            ax.set_ylim([0, calculate_df['log10(p)'].max() + 2])  # Adjusting the range of p-values

            # Find the node with the largest absolute value of log10(p)
            top_points = significant_points.loc[significant_points['log10(p)'].nlargest(node_num).index]
            x = []
            y = []
            labels = []
            for index, row in top_points.iterrows():
                x.append(row['log2FC'])
                y.append(row['log10(p)'])
                labels.append(index)
            scatter = ax.scatter(x, y)
            texts = [ax.text(xi, yi, label, ha='center', va='center') for xi, yi, label in zip(x, y, labels)]
            adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r'))
            # Save or display images
            plt.savefig(os.path.join(save_path, f'{group_arr[i]}-{group_arr[j]}_volcano.pdf'))


def box(file_path, gen_id, save_path='./'):
    if os.path.isabs(save_path):
        save_path = os.path.abspath(save_path)
    if not os.path.isabs(file_path):
        file_path = os.path.abspath(file_path)

    df = pd.read_csv(file_path, sep='\t', low_memory=False)
    df, num_gsm, group_df, group_num, group_arr = read_data(df)

    group_data = [[] for _ in range(len(group_arr))]
    colors = ['skyblue', 'lightgreen', 'salmon', 'lightgrey', 'lightpink', 'lightcoral', 'lightblue', 'lightyellow']

    for i in range(group_num):
        group_data[i] = group_df[i].loc[gen_id, :]

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (data_group, color, label) in enumerate(zip(group_data, colors, group_arr)):
        bplot = ax.boxplot(data_group, positions=[idx + 1], patch_artist=True, widths=0.6)
        plt.setp(bplot['boxes'], facecolor=color, alpha=0.7)  # Case Colour
        plt.setp(bplot['medians'], color='k')  # Median colour
    ax.set_xticks(range(1, len(group_data) + 1))
    ax.set_xticklabels(group_arr)
    plt.title(f'Box Plot({gen_id})')
    plt.ylabel('Value')
    y_annotation_base = plt.ylim()[1] * 1.1
    sep = 0.2 * y_annotation_base
    y_max = plt.ylim()[1] * 2
    ax.set_ylim(0, y_max)
    for i in range(group_num):
        for j in range(i + 1, group_num):
            t_statistic, p_value = ttest_ind(group_data[i], group_data[j])
            fc_value = (sum(group_data[i]) / len(group_data[i])) / (sum(group_data[j]) / len(group_data[j]))
            mid_x = (i + j) / 2 + 1

            ax.axhline(y=y_annotation_base + (j - i) * sep, xmin=(i + 0.5) / len(group_arr),
                       xmax=(j + 0.5) / len(group_arr), color='b',
                       linestyle='--')

            ax.axvline(x=i + 1, ymin=(max(group_data[i]) + sep * 0.2) / y_max,
                       ymax=(y_annotation_base + (j - i) * sep) / y_max, color='b', linestyle=':')
            ax.axvline(x=j + 1, ymin=(max(group_data[j]) + sep * 0.2) / y_max,
                       ymax=(y_annotation_base + (j - i) * sep) / y_max, color='b', linestyle=':')

            plt.text(mid_x, y_annotation_base + (j - i) * sep, f"p={p_value:.2e}, FC={fc_value:.2f}",
                     horizontalalignment='center', verticalalignment='bottom', fontsize=16)

    # Restructuring of the layout
    plt.tight_layout()

    # Show charts
    plt.savefig(os.path.join(save_path, f'box({gen_id}).pdf'))


def heatmap(file_path, save_path='./'):
    if os.path.isabs(save_path):
        save_path = os.path.abspath(save_path)
    if not os.path.isabs(file_path):
        file_path = os.path.abspath(file_path)
    df = pd.read_csv(file_path, sep='\t', low_memory=False)
    df, num_gsm, group_df, group_num, group_arr = read_data(df)
    df = None

    for i in range(group_num):
        for j in range(i + 1, group_num):
            significant_points = get_signpoint(group_df[i], group_df[j], num_gsm)
            significant_points.loc[:, 'sort_order'] = np.where(significant_points['log2FC'] > 1, 1,
                                                               np.where(significant_points['log2FC'] < -1, -1, 0))

            positive_points = significant_points[significant_points['sort_order'] == 1]
            negative_points = significant_points[significant_points['sort_order'] == -1]

            if len(positive_points) > 30:
                positive_points = positive_points.nlargest(30, 'log2FC')
                positive_points = positive_points.sort_values(by='log2FC', ascending=True)
            if len(negative_points) > 30:
                negative_points = negative_points.nsmallest(30, 'log2FC')
                negative_points = negative_points.sort_values(by='log2FC', ascending=True)

            significant_points = pd.concat(
                [positive_points, negative_points, significant_points[significant_points['sort_order'] == 0]])

            df_sign = significant_points.drop(['sort_order', 'log2FC', 'log10(p)', 'condition1', 'condition2'], axis=1)

            rows, cols = df_sign.shape
            figsize_width = min(cols / 3.0, 20)
            figsize_height = min(rows / 5.0, 10)

            fig, ax_heatmap = plt.subplots(figsize=(figsize_width, figsize_height + 2))
            g = sns.heatmap(df_sign, cmap='coolwarm', linewidths=.5, ax=ax_heatmap, cbar=False)

            ax_heatmap.yaxis.tick_right()
            ax_heatmap.yaxis.set_label_position("right")
            ax_heatmap.yaxis.set_label_coords(-0.1, 0.5)
            yticklabels = g.get_yticklabels()
            g.set_yticklabels(yticklabels, rotation=0, va='center', ha='left', fontsize=10)
            for tick in ax_heatmap.get_yticklines():
                tick.set_visible(False)
            ax_heatmap.tick_params(axis='y', which='both', direction='out', length=10)
            ax_heatmap.set_ylabel("")

            # Setting the title above ax_bars1
            title = f"Heatmap of {group_arr[i]} vs {group_arr[j]}"
            fig.suptitle(title, fontsize=20, weight='bold', y=1)

            divider = make_axes_locatable(ax_heatmap)

            # Add top bar (ax_bars1) for group indicator, closer to the heatmap
            ax_bars1 = divider.append_axes("top", size="5%", pad=0.1, sharex=ax_heatmap)
            ax_bars1.barh(0.5, len(group_df[i].columns), height=1, left=0, color='green', edgecolor='none')
            ax_bars1.barh(0.5, len(group_df[j].columns), height=1, left=len(group_df[i].columns), color='blue',
                          edgecolor='none')
            ax_bars1.set_xticks([])
            ax_bars1.set_yticks([0.5])
            ax_bars1.set_yticklabels(['Group'], fontsize=10)
            ax_bars1.yaxis.tick_right()
            ax_bars1.yaxis.set_label_position("right")
            ax_bars1.spines['top'].set_visible(False)
            ax_bars1.spines['right'].set_visible(False)
            ax_bars1.spines['left'].set_visible(False)
            ax_bars1.spines['bottom'].set_visible(False)

            # Add right bar (ax_bars3) for group labels, moved further to the right
            ax_bars3 = divider.append_axes("right", size="10%", pad=1.2, sharey=ax_heatmap)
            ax_bars3.barh(0.5, 0.5, height=1, left=0, color='green', edgecolor='none')
            ax_bars3.barh(1.5, 0.5, height=1, left=0, color='blue', edgecolor='none')
            ax_bars3.axis("off")
            ax_bars3.text(0.75, 0.5, group_arr[i], ha='center', va='center', fontsize=10)
            ax_bars3.text(0.75, 1.5, group_arr[j], ha='center', va='center', fontsize=10)
            # Colorbar below ax_bars3 (width set to half of ax_bars3's width)
            cax = fig.add_axes(
                [ax_bars3.get_position().x1 - 0.02, ax_heatmap.get_position().y1 - ax_bars3.get_position().height / 1.5,
                 0.02, ax_heatmap.get_position().height / 2])
            ColorbarBase(cax, cmap=plt.cm.coolwarm, orientation='vertical')
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
            plt.savefig(os.path.join(save_path, f'heatmap_{group_arr[i]}_{group_arr[j]}.pdf'), bbox_inches='tight')
            plt.close(fig)


def clustering(file_path, if_text=False, n_clusters=3, save_path='./'):
    if os.path.isabs(save_path):
        save_path = os.path.abspath(save_path)
    if not os.path.isabs(file_path):
        file_path = os.path.abspath(file_path)

    df = pd.read_csv(file_path, sep='\t', low_memory=False)
    df = df.drop(df.columns[1], axis=1)
    df = df.drop(df.columns[1], axis=1)
    df.drop(index=0, inplace=True)
    df.set_index(df.columns[0], inplace=True)
    df = df.astype('float').T

    df = cluster_calculate(df, n_clusters)

    # Dimensionality reduction using UMAP for visualisation purposes
    reducer = umap.UMAP(n_components=2)
    umap_components = reducer.fit_transform(df.iloc[:, :-1])
    umap_df = pd.DataFrame(data=umap_components, columns=['UMAP 1', 'UMAP 2'], index=df.index)
    umap_df['Cluster'] = df['Cluster']

    # Visualising clustering results using UMAP
    markers = ['o', 's', '^', 'p', '*', 'x', 'D']  # Increase or decrease the list of shapes according to the number of clusters you have
    unique_clusters = umap_df['Cluster'].unique()
    unique_clusters = sorted(unique_clusters)
    marker_map = {cluster: markers[i % len(markers)] for i, cluster in enumerate(unique_clusters)}

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(right=0.82)
    for cluster in unique_clusters:
        subset = umap_df[umap_df['Cluster'] == cluster]
        ax.scatter(subset['UMAP 1'], subset['UMAP 2'], label=f'Cluster {cluster}',
                   marker=marker_map[cluster], s=100, alpha=0.6)
    if if_text:
        texts = [plt.text(x, y, label, fontsize=8, ha='center', va='center') for x, y, label in
                 zip(umap_df['UMAP 1'], umap_df['UMAP 2'], umap_df.index)]
        adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

    ax.set_title('UMAP Visualization of Sample Clustering with Different Shapes per Cluster', fontsize=16)
    ax.set_xlabel('UMAP Dimension 1', fontsize=14)
    ax.set_ylabel('UMAP Dimension 2', fontsize=14)
    ax.legend(title='Cluster', bbox_to_anchor=(0.9, 0.9), loc='right')  # Move the legend to the outside of the chart

    plt.savefig(os.path.join(save_path, 'clustering.pdf'))


def enrichment(file_path, save_path='./'):
    if os.path.isabs(save_path):
        save_path = os.path.abspath(save_path)
    if not os.path.isabs(file_path):
        file_path = os.path.abspath(file_path)
    save_path = os.path.abspath(save_path)
    warnings.filterwarnings("ignore", category=FutureWarning)

    df = pd.read_csv(file_path, sep='\t', low_memory=False)
    dict_from_df = df.iloc[:, :2].set_index(df.columns[0])[df.columns[1]].to_dict()
    df, num_gsm, group_df, group_num, group_arr = ench_read_data(df)
    for i in range(group_num):
        for j in range(i + 1, group_num):
            significant_points = get_signpoint(group_df[i], group_df[j], num_gsm)

            index_arr = significant_points.index.tolist()
            gene_list = []
            for index in index_arr:
                value = str(dict_from_df[index])
                if value != 'nan':
                    gene_list.append(dict_from_df[index])
            gene_list = list(gene_list)  # 集合强转成数组
            print(gene_list)

            gene_sets = 'KEGG_2019_Human'
            enr = gp.enrichr(gene_list=list(gene_list),  # Required query gene_list, either a list or a file (one column, one gene per row)
                             gene_sets=gene_sets,  # gene set library, multiple related gene sets. For example, all GO terms form a gene set library.
                             organism='Human',
                             outdir=os.path.join(save_path, 'enrichr'),  # output directory
                             top_term=20,
                             cutoff=0.5  # pvalue阈值
                             )
            dot_png = "KEGG_2019" + "_" + "dot" + ".pdf"
            bar_png = "KEGG_2019" + "_" + "bar" + ".pdf"
            base_path = os.path.join(save_path, "static/images/")
            # Creating a directory before saving images
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if not os.path.exists(base_path):
                os.makedirs(base_path)

            if os.path.exists(base_path + dot_png):
                os.remove(base_path + dot_png)
            dotplot(enr.results.loc[enr.results["Gene_set"] == "KEGG_2019_Human",], title='KEGG Top20 Pathway',
                    cmap='viridis_r',
                    top_term=20, legend="r",
                    ofname=base_path + dot_png,
                    )
            # bar
            if os.path.exists(base_path + bar_png):
                os.remove(base_path + bar_png)
            barplot(enr.res2d, title='KEGG_2019', top_term=20, ofname=base_path + bar_png)


def similarity(file_path, gene_a, gene_b, save_path='./'):
    # Make sure the save path is an absolute path
    if not os.path.isabs(save_path):
        save_path = os.path.abspath(save_path)

    # 确保文件路径是绝对路径
    if not os.path.isabs(file_path):
        file_path = os.path.abspath(file_path)

    # Read the data and delete the second and third columns
    df = pd.read_csv(file_path, sep='\t', low_memory=False)
    df = df.drop(df.columns[[0, 2]], axis=1)

    # Assuming that the first column is the gene name, find the whole rows of gene_a and gene_b according to the name in the first column.
    first_column_name = df.columns[0]  # 动态获取第一列的名称
    x = df.loc[df[first_column_name] == gene_a].iloc[0, 1:].astype(float).values
    y = df.loc[df[first_column_name] == gene_b].iloc[0, 1:].astype(float).values

    # Calculate Pearson's correlation coefficient and p-value
    cc, p_value = pearsonr(x, y)

    # Setting up the drawing
    plt.figure(figsize=(8, 6))

    # Scatter plot, colour is light red to avoid too bright colours
    plt.scatter(x, y, color='pink', edgecolor='k', label='Data points', alpha=0.7)

    # fitted line
    plt.plot(x, cc * (x - x.mean()) + y.mean(), color='blue', label=f'Fit line: CC={cc:.3f}')

    # Adjust the axis range to be slightly larger than the data range
    x_min, x_max = x.min() * 0.9, x.max() * 1.1
    y_min, y_max = y.min() * 0.9, y.max() * 1.1
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Add relevance information text, p-value in two decimals and in scientific notation, centred
    plt.text(0.5, 0.95, f'p={p_value:.2e}, CC={cc:.3f}', transform=plt.gca().transAxes, fontsize=12, ha='center')
    plt.text(0.05, 0.05, 'CC: Pearson correlation coefficient', transform=plt.gca().transAxes, fontsize=10)

    # Setting up axis labels and titles
    plt.xlabel(f'{gene_a} expression level (Log2) in cancer', fontsize=12)
    plt.ylabel(f'{gene_b} expression level (Log2) in cancer', fontsize=12)
    plt.title(f'Correlation between {gene_a} and {gene_b} expression levels')

    # Save as PDF
    plt.savefig(os.path.join(save_path, "scatter_plot.pdf"), format="pdf")


def main():
    # Parsing command line arguments
    parser = argparse.ArgumentParser(description="Gene Expression Analysis Tool")

    # Add parameters
    parser.add_argument('method', choices=['volcano', 'box', 'heatmap', 'clustering', 'enrichment', 'similarity'],
                        help="Analysis method to perform")
    parser.add_argument('file_path', help="Path to the input file (CSV format)")
    parser.add_argument('--save_path', default='./', help="Path to save the output (default: current directory)")
    parser.add_argument('--node_num', type=int, default=10, help="Number of nodes for volcano plot (default: 10)")
    parser.add_argument('--gen_id', type=str, help="Gene ID for box plot (required for box plot)")
    parser.add_argument('--gen_a', type=str, help="Gene  for similarity plot ")
    parser.add_argument('--gen_b', type=str, help="Gene  for similarity plot ")

    parser.add_argument('--n_clusters', type=int, default=3, help="Number of clusters for clustering (default: 3)")
    parser.add_argument('--if_text', action='store_true', help="If text labels should be added to the clustering plot")

    # parsing parameter
    args = parser.parse_args()

    # Check if the save path is an absolute path
    if not os.path.isabs(args.save_path):
        args.save_path = os.path.abspath(args.save_path)

    # Calling different functions depending on the method
    if args.method == 'volcano':
        volcano(args.file_path, args.save_path, args.node_num)
    elif args.method == 'box':
        if not args.gen_id:
            print("Error: 'gen_id' is required for box plot.")
            return
        box(args.file_path, args.gen_id, args.save_path)
    elif args.method == 'heatmap':
        heatmap(args.file_path, args.save_path)
    elif args.method == 'clustering':
        clustering(args.file_path, if_text=args.if_text, n_clusters=args.n_clusters, save_path=args.save_path)
    elif args.method == 'enrichment':
        enrichment(args.file_path, args.save_path)
    elif args.method == 'similarity':
        similarity(args.file_path, args.gen_a, args.gen_b, args.save_path)


if __name__ == '__main__':
    main()