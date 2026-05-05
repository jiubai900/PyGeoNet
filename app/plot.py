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
from .plot_tool import read_data, cluster_calculate, get_signpoint, get_point, benjamini_hochberg
from matplotlib.colorbar import ColorbarBase


def volcano_plot(file_path, save_path='./'):
    df = pd.read_csv(file_path, sep='\t', low_memory=False)
    df, num_gsm, group_df, group_num, group_arr = read_data(df)

    for i in range(group_num):
        for j in range(i + 1, group_num):

            significant_points, non_significant_points, point, calculate_df = get_point(group_df[i], group_df[j], num_gsm)

            # 将显著点进行输出
            df_result = significant_points[['log10(p)', 'log2FC']]
            df_result = df_result.reset_index()
            df_result.columns = ['gene', 'p-value', 'Fold Change']
            p_value = df_result['p-value'].values
            adjusted_p_values = benjamini_hochberg(p_value)
            df_result['FDR'] = adjusted_p_values
            file_name = group_arr[i] + '_' + group_arr[j] + '.txt'
            df_result.to_csv(os.path.join(save_path, file_name), sep='\t', index=False)

            # 绘制图像
            fig, ax = plt.subplots(figsize=(8, 6))
            plt.axvline(x=0, color=".5")
            plt.axhline(y=-np.log10(0.05), linestyle='--')
            plt.axvline(x=1, linestyle='--')  # 绿色虚线
            plt.axvline(x=-1, linestyle='--')  # 绿色虚线
            ax.scatter(calculate_df['log2FC'], calculate_df['log10(p)'], color='grey')  # 所有点
            ax.scatter(significant_points['log2FC'], significant_points['log10(p)'], color='red', label='Significant '
                                                                                                        'Points')  # 显著点
            ax.scatter(non_significant_points['log2FC'], non_significant_points['log10(p)'], color='blue')  # 非显著点
            ax.scatter(point['log2FC'], point['log10(p)'], color='green')  # 非显著点
            # 添加轴标签和标题
            ax.set_xlabel('Log2 Fold Change')
            ax.set_ylabel('-log10(p-value)')
            ax.set_title(f'{group_arr[i]}-{group_arr[j]} Volcano Plot')

            # 添加图例
            ax.legend()

            # 可选：设置x和y轴的限制
            ax.set_xlim([calculate_df['log2FC'].min() - 1, calculate_df['log2FC'].max() + 1])  # 调整FC的范围
            ax.set_ylim([0, calculate_df['log10(p)'].max() + 2])  # 调整p值的范围

            top_points = calculate_df.nlargest(10, 'log10(p)')
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
            # 保存或显示图像
            plt.savefig(os.path.join(save_path, f'{group_arr[i]}_{group_arr[j]}volcano.png'))


def box_plot(file_path, gen_id, save_path='./'):
    df = pd.read_csv(file_path, sep='\t', low_memory=False)
    df, num_gsm, group_df, group_num, group_arr = read_data(df)

    group_data = [[] for _ in range(len(group_arr))]
    colors = ['skyblue', 'lightgreen', 'salmon', 'lightgrey', 'lightpink', 'lightcoral', 'lightblue', 'lightyellow']

    for i in range(group_num):
        group_data[i] = group_df[i].loc[gen_id, :]

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (data_group, color, label) in enumerate(zip(group_data, colors, group_arr)):
        bplot = ax.boxplot(data_group, positions=[idx + 1], patch_artist=True, widths=0.6)
        plt.setp(bplot['boxes'], facecolor=color, alpha=0.7)  # 箱体颜色
        plt.setp(bplot['medians'], color='k')  # 中位数颜色
    ax.set_xticks(range(1, len(group_data) + 1))
    ax.set_xticklabels(group_arr)
    plt.title('Box Plot')
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
                     horizontalalignment='center', verticalalignment='bottom', fontsize=15)

    # 调整布局
    plt.tight_layout()

    # 显示图表
    plt.savefig(os.path.join(save_path, 'box.png'))


def heatmap(file_path, save_path='./'):
    df = pd.read_csv(file_path, sep='\t', low_memory=False)
    df, num_gsm, group_df, group_num, group_arr = read_data(df)
    df = None
    for i in range(group_num):
        for j in range(i + 1, group_num):
            significant_points = get_signpoint(group_df[i], group_df[j], num_gsm)
            significant_points.loc[:, 'sort_order'] = np.where(significant_points['log2FC'] > 1, -1, 1)
            significant_points = significant_points.sort_values(by=['sort_order', 'log2FC'])
            df_sign = significant_points.drop(['sort_order', 'log2FC', 'log10(p)', 'condition1', 'condition2'], axis=1)
            # 热力图的数据
            rows, cols = df_sign.shape
            figsize_width = 40  # 防止宽度过大
            figsize_height = min(rows / 10.0, 70)  # 防止高度过大
            # 基础热图
            fig, ax_heatmap = plt.subplots(figsize=(figsize_width, figsize_height))
            g = sns.heatmap(df_sign, cmap='coolwarm', linewidths=.5, ax=ax_heatmap, cbar=False)
            ax_heatmap.yaxis.tick_right()
            ax_heatmap.yaxis.set_label_position("right")
            ax_heatmap.yaxis.set_label_coords(-0.1, 0.5)
            yticklabels = g.get_yticklabels()
            g.set_yticklabels(yticklabels, rotation=0, va='center', ha='left', fontsize=10)
            for tick in ax_heatmap.get_yticklines():
                tick.set_visible(False)  # 隐藏默认的y轴刻度线
            ax_heatmap.tick_params(axis='y', which='both', direction='out', length=10)  # 创建新的向外的y轴刻度线
            ax_heatmap.set_ylabel("")

            divider = make_axes_locatable(ax_heatmap)

            # 增加上方子图-父图标
            ax_bars1 = divider.append_axes("top", size=0.5, pad=0.1, sharex=ax_heatmap)

            ax_bars1.barh(1 / 2, len(group_df[i].columns), height=1, left=0, align='center', color='green',
                          edgecolor='none')
            ax_bars1.barh(1 / 2, len(group_df[j].columns), height=1, left=1 * len(group_df[i].columns), align='center',
                          color='blue', edgecolor='none')

            # 隐藏子图的x轴刻度和标签
            ax_bars1.set_xticks([])
            ax_bars1.xaxis.set_ticklabels([])
            ax_bars1.set_yticks([0.5])
            ax_bars1.set_yticklabels(['Group'])
            ax_bars1.yaxis.tick_right()
            ax_bars1.yaxis.set_label_position("right")

            # 增加子图——右边图例

            ax_bars3 = divider.append_axes("right", size=0.5, pad=2, sharey=ax_heatmap)
            ax_bars3.barh(0.5, 0.5, height=1, left=0, align='center', color='green',
                          edgecolor='none')
            ax_bars3.barh(1.5, 0.5, height=1, left=0, align='center',
                          color='blue', edgecolor='none')
            ax_bars3.axis("off")
            ax_bars3.text(0.8, 0.5, group_arr[i], ha='center', va='center')
            ax_bars3.text(0.8, 1.5, group_arr[j], ha='center', va='center')
            ax_bars3.set_ylabel('group')
            ax_bars3.set_title('group', fontsize=12)  # 自定义你的标题内容和字体大小

            # 获取'coolwarm'颜色映射
            cmap = plt.cm.coolwarm
            # 创建一个颜色条轴并绘制与热图相同颜色映射
            # 注意调整cax的位置，使其位于热图右侧并适当偏移避免重叠
            cax = fig.add_axes([ax_heatmap.get_position().x1 - 0.03, ax_heatmap.get_position().y0, 0.02,
                                ax_heatmap.get_position().height])
            ColorbarBase(cax, cmap=cmap, orientation='vertical')

            plt.tight_layout()
            # 显示热图
            plt.savefig(os.path.join(save_path, f'heatmap_{group_arr[i]}_{group_arr[j]}.png'))


def clustering(file_path, if_text=False, n_clusters=3, save_path='./'):
    df = pd.read_csv(file_path, sep='\t', low_memory=False)
    df = df.drop(df.columns[1], axis=1)
    df = df.drop(df.columns[1], axis=1)
    df.drop(index=0, inplace=True)
    df.set_index(df.columns[0], inplace=True)
    df = df.astype('float').T

    df = cluster_calculate(df, n_clusters)

    # 使用UMAP降维以便可视化
    reducer = umap.UMAP(n_components=2)
    umap_components = reducer.fit_transform(df.iloc[:, :-1])
    umap_df = pd.DataFrame(data=umap_components, columns=['UMAP 1', 'UMAP 2'], index=df.index)
    umap_df['Cluster'] = df['Cluster']

    # 可视化聚类结果使用UMAP
    markers = ['o', 's', '^', 'p', '*', 'x', 'D']  # 根据你的聚类数量增减形状列表
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

    ax.set_title('UMAP Visualization of Sample Clustering with Different Shapes per Cluster')
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')  # 将图例移到图表外侧

    plt.savefig(os.path.join(save_path, 'clustering.png'))


def enrichment_plot(file_path, save_path='./'):
    warnings.filterwarnings("ignore", category=FutureWarning)

    df = pd.read_csv(file_path, sep='\t', low_memory=False)
    dict_from_df = df.iloc[:, :2].set_index(df.columns[0])[df.columns[1]].to_dict()
    df, num_gsm, group_df, group_num, group_arr = read_data(df)
    for i in range(group_num):
        for j in range(i + 1, group_num):
            significant_points = get_signpoint(group_df[i], group_df[j], num_gsm)

            index_arr = significant_points.index.tolist()
            gene_list = {}
            for index in index_arr:
                value = str(dict_from_df[index])
                if value != 'nan':
                    gene_list.add(dict_from_df[index])
            gene_list = list(gene_list)  # 集合强转成数组

            gene_sets = 'KEGG_2019_Human'
            enr = gp.enrichr(gene_list=gene_list,  # 所需查询gene_list，可以是一个列表，也可为文件（一列，每行一个基因）
                             gene_sets=gene_sets,  # gene set library，多个相关的gene set 。如所有GO term组成一个gene set library.
                             organism='Human',  # 持(human, mouse, yeast, fly, fish, worm)， 自定义gene_set 则无影响。
                             outdir=os.path.join(save_path, 'enrichr'),  # 输出目录
                             top_term=20,
                             cutoff=0.5  # pvalue阈值
                             )
            dot_png = "KEGG_2019" + "_" + "dot" + ".png"
            bar_png = "KEGG_2019" + "_" + "bar" + ".png"
            base_path = "./static/images/"
            # 在保存图片之前创建目录
            if not os.path.exists(save_path):
                os.makedirs(save_path)

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


# # volcano_plot("C:\\Users\\1\\Desktop\\mRNA(2)\\GSE6280\\GSE6280-GPL96.txt", 'D:\\GEO_data')
# heatmap("C:\\Users\\1\\Desktop\\mRNA(2)\\GSE6280\\GSE6280-GPL96.txt")
# clustering("C:\\Users\\1\\Desktop\\mRNA(2)\\GSE6280\\GSE6280-GPL96.txt",True)
# box_plot("C:\\Users\\1\\Desktop\\mRNA(2)\\GSE6280\\GSE6280-GPL96.txt", '226228_at')