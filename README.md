# PyGeoNet: Python toolkit for search and analysis based on the GEO database

## Introduction
`PyGeoNet`is a Python toolkit designed to simplify the process of searching, downloading, analysing and visualising mRNA expression data from the GEO (Gene Expression Omnibus) database. It provides researchers with convenient and fast functions for searching, downloading, pre-processing and analysing gene expression data, enabling users to easily process and analyse large-scale genomic data.

## Installation
```bash
git clone https://github.com/jiubai900/PyGeoNet.git
cd PyGeoNet
# Installation of dependencies
pip install -r requirements.txt
# Note: PyTorch needs to be installed separately according to the CUDA version (see the CUDA configuration of the model module for details)
```


## Overall workflow
The core process of PyGeoNet is divided into 6 steps from GEO data acquisition to pathway analysis, with each step relying on the output of the previous step:
1. **Data search** (`search.py`): filter GSE numbers based on keywords to get a list of research projects.
2. **Data download** (`download.py`): download raw data (soft/matrix/GPL etc.) according to GSE number.
3. **Data normalisation** (`normalize.py`): process raw data to generate gene expression matrices.
4. **Difference analysis** (`analyse.py`): generates visualisations (volcano/heatmaps etc.) after manual grouping of expression matrices.
5. **Model preprocessing** (`standard.py`): calculates gene similarity and generates graph model input format (adjacency matrix/feature matrix).
6. **Model training and prediction** (`model.py`): train graph neural networks to predict genetic relationships.
7. **Pathway analysis** (`pathway.py`): visualise genetic differences within/between pathways based on predictions.



## basic usage
-----
### search.py— Data search module

Search for GSE projects in GEO based on keywords and get detailed meta information with support for exporting to Excel files.

----

#### **function list** ：
- `gse`

    - **functionality**：Search based on multiple field conditions. Supports input of multiple filters and returns a list of matching GSEs for subsequent data download and analysis.

    - **function parameter**
        - --ALL: (str)，All terms in all searchable fields, such as：`'cancer'`
        - --AUTH: (str)，Contributors or authors associated with the study, such as：`'smith a'`
        - --GTYP: (str)，DataSet or Series type, e.g.：`'expression profiling by high throughput sequencing'`
        - --DESC: (str)，Text provided in DataSet, Series, or Sample description, summary, and other metadata fields, such as：`'smok*'`
        - --MESH: (str)，Medical Subject Headings (MeSH) terms, such as：`'methylation methylation'`
        - --NPRO: (int)，Number of platform probe IDs, e.g.：`'1000000:1000000000'`
        - --NSAM: (int)，The number of samples in the DataSet or Series, for example：`'100：500'`
        - --ORGN: (str)，The name of the organism, e.g.：`'Homo sapiens'`
        - --PDAT: (str)，The name of the organism, e.g.：`'2007/01：2007/06'`
        - --RGPL: (str)，Retrieves the Plaform of the specified DataSet or Series, for example：`'GPL570'`
        - --SRC: (str)，The source of the biological material of the sample, e.g.：`'brain'`
        - --VTYP: (str)，Sample value type, e.g.：`'log ratio'`
        - --TITL: (str)，Text in DataSet, Series, Platforms, and Samples headers, such as：`'Affymetrix'`
        - --UDAT: (str)，The date when the record was last updated, e.g.：`'2010/06'`
        - --retmax: (int)，Specify the maximum number of records to be returned, the default is`5000`。

        - For detailed naming rules, please refer to  https://www.ncbi.nlm.nih.gov/geo/info/qqtutorial.html


- `contents`

    - **functionality** ：Crawl relevant GEO page information based on GSE lists. Supports multi-threaded crawling and returns an Excel file with detailed information about the project.

    - **function parameter**
        - `--gse_arr`: (str) a list of GSE numbers, usually returned by the gse function。
        - `--save_path`: (str) Path to save capture results。
        - `--output_format`: (str) Output formats, csv, xlsx, df, txt are supported.。
        - `--thread`: (int) the number of threads to use when crawling, default is `8`, it is recommended that the number of threads is not too large, because if there are too many threads, it may cause the server to crash, resulting in a large number of inaccessible problems。

------

#### example code
```bash

# get all the keywords in line with the GSE, the results will be stored in search.txt to facilitate the second part of the call, because the second part of the call will often appear in response to the case of timeout, the need for multiple attempts, so it is necessary to carry out data preservation
python search.py gse --ALL 'ccRCC'  'renal cancer'  'renal tumor'  'PRCC'  'ChRCC'  'kidney cancer'  'kindney tumor'  'renal cell' --NSAM '5:10000' --ORGN "Homo sapiens"  > search.txt

# Getting the corresponding information through GSE returns a file called GEO.xlsx
# Call the results of the previous step to search
python search.py contents --gse_arr './search.txt' --save_path './' 
# Direct calls to specific GSEs for searching
python search.py contents --gse_arr 'GSE11151' 'GSE53757' --save_path './'

```
------

### download.py — Data Download Module
`download`The module provides multi-threaded download of GEO data, supports download of soft, matrix, GPL and suppl files, and has the function of repeating failed download files。

-----

#### **function list** 
- `soft`
    - **functionality**：Download `soft` files corresponding to GSEs in the GEO database. Supports input of a list of GSE numbers or an Excel file。
    - **function parameter**

        - `--gse_arr`: GSE number (string, list or Excel file)。
        - `--save_path`: Path to save the downloaded file。
        - `--log_path`: Log file path。
        - `--thread`: Multi-threaded download, default`4`。

- `matrix`
    - **Function**: Download `matrix` files corresponding to GSEs in the GEO database. Supports input of a list of GSE numbers or an Excel file.。

    - **function parameter**

        - `--gse_arr`: GSE number (string, list or Excel file)。
        - `--save_path`: Path to save the downloaded file。
        - `--log_path`: Log file path.
        - `--thread`: Multi-threaded download, default`4`。

- `suppl`
    - Function: Download `suppl` file corresponding to GSE in GEO database. Supports input of a list of GSE numbers or an Excel file.。

    - **function parameter**

        - `--gse_arr`: GSE number (string, list or Excel file)。
        - `--save_path`: Path to save the downloaded file。
        - `--log_path`: Log file path。
        - `--thread`: Multi-threaded download, default`4`。


- `gpl`
    - **functionality**: Download gpl (platform files) from the GEO database. Support for entering GSE number lists or Excel files。

    - **function parameter**

        - `--gse_arr`: A list of platform numbers, either as a string, a list, or as a path to an Excel file, with the Platform column in the Excel file.。
        - `--save_path`: Path to save the downloaded file。
        - `--log_path`: Log file path。
        - `--thread`: Multi-threaded download, default`4`。

- `all`
    - **functionality**: Download all files corresponding to GSEs in the GEO database at once, including soft, matrix and suppl files, with other parameters and behaviour identical to the download function for soft files.

    - **function parameter**

        - `--gse_arr`: GSE number (string, list or Excel file)。
        - `--save_path`: Path to save the downloaded file。
        - `--log_path`: Log file path。
        - `--thread`: Multi-threaded download, default`4`。

- `down_again`
    - **Function**: Repeat downloads for previously failed files. Automatically re-downloads the unfinished part by reading the log file of the previous error.。

    - **Function parameters**

        - `--log_path`: A log file generated during the previous download containing information about download failures (can be soft, matrix, suppl, gpl logs).
        - `--save_path`: The path where the file is saved needs to be the same as the previous download path.
        - `--thread`: Number of download threads, default `4`.
    - Implementation process:
        1. Failed downloads are re-downloaded; successful downloads are not repeated.
        2. A new log file is generated after the download until all files have been successfully downloaded.

-------

#### **sample code (computing)**
```bash
# Download soft, matrix, suppl, all formats are the same, just replace the contents of command
# Method one
python download.py all --gse_arr './GEO.xlsx' --save_path './DATA/download' --log_path './DATA/log'
# Method Two
python download.py all --gse_arr 'GSE11151' 'GSE247118' --save_path './DATA/download' --log_path './DATA/log'

# Download gpl file
# Method one
python download.py gpl --gse_arr './GEO.xlsx' --save_path './DATA/GPL' --log_path './DATA/log'
# Method Two
python download.py gpl --gse_arr 'GPL570' 'GPL97' --save_path './DATA/GPL' --log_path './DATA/log'

# Re-downloading failed files
python download.py down_again --save_path './DATA/download' --log_path './DATA/log/2024-11-01-suppl-logger.log'
 

```
-------

### normalize.py — Expression matrix standardisation module
`normalize`The module is mainly used to analyse and process the downloaded raw data to generate a standardised mRNA expression data format that can be directly used by subsequent modules.

------

#### **function list**  
- `get_data`

    - **functionality**: This function will download the original data for complex processing, get the gene and sample mRNA expression data. It supports processing mRNA data in some common formats, but some uncommon formats are not supported for the time being.。

    - **function parameter**
        - `raw_data_path`: The path to the raw data, usually the path to the data folder containing the `soft`, `matrix`, and `suppl` files downloaded via the `download` module.
        - `--gpl_path`: The path to the platform file (optional). If not provided, the system will automatically download the required GPL platform data, but this may take extra time. The default value is the empty string ''.
        - `--save_path`: The path to save the processed mRNA expression data. The default value is `'. /''.
        - `--thread`: The number of threads to use when processing data, default value is `4`.
        - `--single_operation`: Whether to process only single files. If the folder contains multiple GSE filesets, batch processing is performed by default (single_operation=False); if only a single GSE fileset needs to be processed, this parameter needs to be entered.

------

#### Implementation process
1. If `gpl_path` is provided, the platform file under the specified path will be used directly for processing; otherwise, the system will automatically download the corresponding platform file.。
2. Depending on the `single_operation` parameter, the function will decide whether to process a single GSE file or multiple file sets.
3. After processing, the normalised mRNA expression data will be saved to the `save_path` specified directory for use in subsequent analysis.

------

#### sample code (computing)
```bash
# Handles multiple GSE files and automatically downloads required GPL files
python normalize.py './DATA/download' --save_path './DATA/normalize' --gpl_path './DATA/GPL'

# Processing of individual GSE files, platform file paths have been provided manually
python normalize.py './DATA/download/GSE11151' --save_path './DATA/normalize' --single_operation
```
------

#### clarification
- The data files in `raw_data_path` need to contain `soft`, `matrix`, and `suppl` files to ensure complete data processing.
- If `gpl_path` is not provided, the system will automatically download the required GPL file, which may take some time.
- `single_operation=True` is used to handle the case of a single fileset and is suitable for manual modification of irregular data before normalisation.

------

###   analyse.py — Variance analysis and visualisation module
`analyse` focuses on differential analysis of mRNA data, with visual plots showing differences between genes in different samples.
It is important to note here that this step needs to be performed on the data processed by the `normalise` module in the previous step.
The main processing is a manual grouping operation and all visualisation results are based on grouping.

------
#### Manual sorting operations (core step, must be completed)
All functions of the `analyse` module depend on the sample grouping information, and you need to manually add grouping labels to the expression matrix generated by `normalize`:

1. **Open the file**: Find the expression data file output by `normalise` module (e.g. `. /DATA/normalize/GSE53757.txt`) with the following file format (example):
```
| GeneID | Sample1 |Sample2 | Sample3 | Sample4|  # First row: Gene ID + Sample ID
| GeneA  |    2.3  |   2.5  |   1.1   |   1.0  |
...
```
2. **Add grouping**: Insert sample grouping labels in the **second line** in the format required:
- Separated by English commas, strictly corresponding to the order of the sample IDs in the first line
- Grouping labels are recommended to be in English (e.g. `Control`/`Treatment`) and not contain spaces or special characters
- Example (adding 2 sets of labels for 4 samples):
```
| GeneID | Sample1 | Sample2 | Sample3 | Sample4 | 
|category| Control | Control |Treatment|Treatment| # Added a second row: grouping labels
| GeneA  |    2.3  |    2.5  |   1.1   |   1.0   |
...
```
----
#### **List of functions**
- `volcano`
    - **Function**: Differential analysis of gene expression data using volcano plots, distinguishing the significance of different genes and labelling the ten genes with the most significant differences.

    - **Function parameters**
        - `--file_path`: `get_data`The expression data file is generated and the file has to be manually classified. The user only needs to change the second line of the data file to the classification result. The volcano plot will show significant differences in genes between samples.
        - `--save_path`: The path to save the volcano map, the default value is the current directory `'. /'`.
        - `--node_num`: Number of significant gene annotations, default value is `10`.

 - `box`
    - **Function**:Generates a box-and-line plot of the expression data for a single gene, which is used to show how the gene varies across samples.

    - **Function parameters**
        - `--file_path`:`get_data` The expression data file generated is subject to manual classification. The second line of the file should be modified to read classification results.
        - `--gen_id`: Number of the gene to be differentially analysed.
        - `--save_path`: The path to save the boxplot, the default value is `'. /'`.

- `heatmap`
    - **Function**: Performs differential analysis of expression data, using heatmaps to show gene expression patterns with significant differences.

    - **Function parameters**
        - `--file_path`:The expression data file generated by `get_data` is subject to manual classification. The second line of the file should be modified to read the results of the classification, and the function will perform a comparison of the differences between the two samples.
        - `--save_path`: The path to save the heatmap, the default value is `'. /'`.

- `clustering`
    - **Function**: Differential analysis of gene expression data by gene clustering map to show the expression pattern of samples in terms of the clustering relationship of different genes.

    - **Function parameters**
        - `--file_path`:Expression data file generated by `get_data`.
        - `--if_text`: Whether to show labels for each gene, default value is `False`.
        - `--n_clusters`: The number of clusters, default value is `3`.

- `enrichment`
    - **Function**: Perform enrichment analysis and graphically display the results of the enrichment analysis.

    - **Function parameters**
        - `--file_path`:Expression data file generated by `get_data`.
        - `--save_path`: The path where the results of the enrichment analysis will be saved, with a default value of `'. /'`.

- `similarity`
    - **Function**: perform correlation analyses and graphically display the results of the correlation between two genes.

    - **Function parameters**
        - `--file_path`: the expression data file generated by `get_data`.
        - `--gen_a`: the first gene to be analysed for correlation.
        - `--gen_b`: the second gene to be analysed for correlation.
        - `--save_path`: correlation analysis result save path, default value is `'. /'`

-----
#### sample code (computing)
```bash
# Group the GSE53757.txt file under the normalise file and store it under the plt folder.

# Generate a volcano map
python analyse.py volcano "./DATA/plt/GSE53757.txt" --save_path './DATA/plt'

# Generate a box plot of the specified genes, common cancer genes are (VHL, PBRM1, SETD2, BAP1, TP53).
python analyse.py box "./DATA/plt/GSE53757.txt" --gen_id 'VHL' --save_path './DATA/plt'

# Generate Heat Map
python analyse.py heatmap "./DATA/plt/GSE53757.txt" --save_path './DATA/plt'

# Generate a clustering diagram
python analyse.py clustering "./DATA/plt/GSE53757.txt" --save_path './DATA/plt' --n_clusters 2 

# Generate enrichment analysis graphs
python analyse.py enrichment "./DATA/plt/GSE53757.txt" --save_path './DATA/plt'

# Generate correlation analysis charts
python analyse.py similarity "./DATA/plt/GSE53757.txt" --save_path './DATA/plt'  --gen_a 'VHL' --gen_b 'PBRM1'
```
------
#### clarification
- In each `file_path` input, the data should be generated via `get_data` and the second row of the data should be manually sorted to ensure proper variance analysis.
- `volcano`, `box`, `heatmap`, `clustering`, `enrichment`, and `similarity` are five common graphs for displaying gene differential analysis in mRNA data, and are suitable for use in a wide range of visualisation needs for gene expression.

-----

### standard.py — Graph model input preprocessing module

`standard`The main function of the module is to pre-process mRNA expression data to generate a standardised input format for subsequent model training.

----

#### function list 
- `similarity`
    - **Function**: Calculates the similarity of gene expression data and outputs gene pairs that meet the similarity threshold.
    - **Function parameters**
        - `--data_path`:`get_data`Path to where the processed dataset is stored. It is required that this data has been manually categorised and split into multiple separate files by category (one file per group).
            For each file, the classification tags need to be removed, leaving only the GSE number as the row index, the gene name as the column name, and the corresponding expression value to ensure format compatibility.
        - `--output_path`: The path where the results of the similarity calculation are saved.
        - `--alpha`: Confidence interval threshold, default value `0.05`.
        - `--sim_threshold`: Similarity filtering threshold, which by default filters out gene pairs with similarity greater than `0.3`.
        - `--num_threads`: The number of threads to use to speed up similarity filtering, with a default value of `1`.
        - `--chunk_size`: The number of rows (data block size) to filter at a time, with a default value of `500`.
- `statistic`
    - **Function**: Statistical screening of similarity calculations and normalisation of gene expression data to generate sparse neighbour matrices, gene translation tables and normalised expression data matrices.
    - **Function parameters**
        - `--sim_data_path`: path to the similarity data file generated by `similarity`.
        - `--expr_data_path`: The `get_data` output expresses the data path, the data should be manually sorted and the second line removed.
        - `--output_path`: Result save path.
        - `--count_threshold`: Threshold for gene pair retention, which indicates the frequency threshold (in per cent) at which a gene pair occurs in all files, with a default value of `0.8`.
- `pca`
    - **Function**: Performs Principal Component Analysis (PCA) on standardised expression data, downscales the data to the specified dimensions and saves the principal component contribution graph.
    - **Function parameters**
        - `--data_path`: Path to the normalised expression data file for `statistic` output.
        - `--png_path`: Saving paths for principal component contribution plots.
        - `--n_components`: The principal component scores in principal component analysis, with a default value of `5`.

----

#### sample code (computing)
```bash
# Calculating similarity
python standard.py similarity --data_path './DATA/standard' --output_path './DATA/sim_result'  --sim_threshold 0.5

# Statistical screening and standardised expression data
python standard.py statistic --sim_data_path './DATA/sim_result' --expr_data_path './DATA/standard' --output_path './DATA/model-data'

# Perform PCA downscaling
python standard.py pca --data_path './DATA/model-data/expression' --png_path './DATA/pca-png' --output_path './DATA/model-data/expression'

```
-----

### model.py — Graph Neural Network Training and Prediction
Mainly used for node classification training, it provides a wide choice of graph neural networks and loss functions.

#### function list
- `train`
    - **Function**: train a node classification model.

    - **Function parameters**
        - `--adj_matrix_path`: The file path of the sparse adjacency matrix.
        - `--feature_matrix_path`: The file path of the feature matrix, with the last column being the label of the node.
        - `--num_classes`: The number of classifications of the node.
        - `--save_path`: Save the path to the training model parameters and output the .pth file.
        - `--epoch`: The number of training iterations, default `200`.
        - `--lr`: Learning rate, default `0.01`.
        - `--model_class`: Model classes, supporting `GAEModel`, `GATModel`, `GCNModel` and so on.
        - `loss_fn`: Loss function with support for `bce_loss`, `mse_loss`, etc.
 
- `predict`
    - **Function**: Make predictions about the data using a trained classification model.
    - **Function parameters**
        - `--adj_matrix_path`: The file path of the sparse adjacency matrix.
        - `--feature_matrix_path`: File path to the feature matrix.
        - `--model_path`: Path to the trained `.pth` model file.
        - `--num_classes`: The number of classifications of the node.
        - `--output_path`: The path to the output result.
        - `--model`: Model class used, default `GAEModel`。

- `edge_pre`
    - **Function**: node updating of graph data by graph neural networks, deletion and addition of edges based on graph similarity to predict genetic relationships.
    - **Function parameters**
        - `--data_path`: Gene pair file path (sparse adjacency matrix, preprocessed by `preprocessing.statistic()`).
        - `--feature_dir`: Path to the preprocessed expression data file (after `statistic()` and `pca()`), file address of the feature matrix.
        - `--gene_map_path`: The file path of the gene translation.
        - `--model_name`:Selection of the underlying graph neural network, with choices of `GAT`, `GCN`, `GraphSAGE` and `ChebNet`
        - `--conv_channels`:The number of features in each layer of the convolution, for example, if there are three layers of convolution you can write [8,16,8], the first layer will make the number of features 8 dimensional and so on.
        - `--th_filter`: Threshold (in per cent) for deletion of edges.
        - `--th_add`: Increase the threshold of the edge (in per cent).
        - `--th_filter_max`: Maximum threshold for deletion of edges, default value is `0.5`.
        - `--th_filter_min`: Minimum threshold for deletion of edges, defaults to `0.01`.
        - `--th_add_max`: Increases the maximum threshold for edges, with a default value of `0.5`.
        - `--th_add_min`: Increases the minimum threshold for edges, with a default value of `0.01`.
        - `--epoch`: The number of iterations, default value is `1`.
----
#### sample code (computing)

```bash

# Training Models
python model.py train --adj_matrix_path './DATA/example/adj_matrix.txt' --feature_matrix_path './DATA/example/feature_matrix_train.csv' --num_class 3 --save_path './DATA/example' --epochs 200 --lr 0.01 --model_class GAEModel --loss_fn bce_loss

# model prediction
python model.py predict --adj_matrix_path './DATA/example/adj_matrix.txt' --feature_matrix_path './DATA/example/feature_matrix_pre.csv' --model_path './DATA/example/model.pth' --num_classes 3 --output_path './DATA/example' --model GAEModel

# Side prediction model
python model.py edge_pre --data_path "./DATA/model-data/gene.txt" --feature_dir './DATA/model-data/expression' --gene_map_path "./DATA/model-data/gene_map.txt" --model_name 'GCN' --conv_channels 8 12 9 --th_filter '0.1' --th_add '0.1'

```

Here you can call cuda to accelerate, because the system environment is not the same so here is the installation tutorial.

```
- Step 1: Check the CUDA version
    Firstly, you need to confirm the version of CUDA installed on your device. You can check it by typing the following command in the terminal:

    ```bash
    nvcc --version
    ```
    If you don't have CUDA installed, or if you're not sure if you have it installed correctly, you can head over to NVIDIA's official website to see how to install the CUDA toolkit, or you can just check for CUDA support for your GPU with the `nvidia-smi` command.
    ```bash
    nvidia-smi
    ```
    It will show CUDA version, GPU utilisation and other information
- Step 2: Choose the right version of PyTorch

    The installation of PyTorch requires you to choose the appropriate version depending on your CUDA version. You can visit the official PyTorch (https://pytorch.org/) website to see the recommended installation commands.
- Step 3: Installation using `pip` or `conda`

    Assuming you have confirmed the CUDA version, you can next install PyTorch using either of the following two methods.
    1. Installation using pip (assuming CUDA 11.7)

        Depending on the version of CUDA you choose, the following commands will automatically download and install the version of PyTorch that supports CUDA for you:
        ```bash
        pip install torch torchvision torchaudio
        ```
        If you are sure that the CUDA version is 11.7, you can specify that:
        ```bash
        pip install torch==2.1.0+cu117 torchvision==0.15.0+cu117 torchaudio==2.1.0+cu117
        ```
        The +cu117 here indicates the CUDA 11.7 supported version. If you have another CUDA version (e.g. 11.6), you need to modify cu117 in the command to the correct CUDA version (e.g. cu116) as needed.
    2. Installation using conda (assuming CUDA 11.7)

        If you are using Anaconda (Conda installation is recommended to avoid package management conflicts), you can run the following command:
        ```bash
        conda install pytorch torchvision torchaudio cudatoolkit=11.7 -c pytorch
        ```
        This will install PyTorch, TorchVision and Torchaudio for CUDA 11.7. If you have a different version of CUDA, simply replace cudatoolkit=11.7 with the appropriate version.
- Step 4: Verify Installation

    Once the installation is complete, you can verify that PyTorch is properly installed and supports CUDA with the following Python code:
    ```python
    import torch
    print(torch.__version__)  # Exporting PyTorch Versions
    print(torch.cuda.is_available())  # Check if CUDA is available
    ```
```

### pathway.py — Visualisation of pathway differences
The results of the `edge_pre` runs were visualised and analysed to show the intra-group differences between pathways in the form of graphs versus inter-group differences.

#### function list 
- `intra`

    - **Function**: Performs within-group analyses based on pathways.
    - **Function parameters**
        - `--pathway_file`: The target pathway, the pathway to be visualised, is formatted with the first column as the pathway name and the rest of the columns as the genes in the pathway, and the file is an excel spreadsheet.
        - `--input_txt`: Target file, the gene pair file to be visualised, in `.txt` format.
        - `--output_file`: The address of the file output, in this case a folder.
        - `--gene_relationship_file`: Filter criteria for gene pairs, this is gene pairs with known relationships, default value is `contrast_gene.txt`.
        - `--max_pathways`: Maximum number of pathways to be displayed, pathways with unique genes will be displayed here, if any of the pathways in the maximum five pathways have all the genes of the pathway contained in the shared genes then they will not be displayed, the default value is `5`.
        - `--figure_size`: Image size, the image size can be adjusted according to the number of genes, the default value is `15`.
        - `--max_pathway_radius`:The maximum radius of the pathway, default value is `15`.
        - `--min_pathway_radius`：The minimum radius of the pathway, with a default value of `2.5`.
        - `--gene_radius`:The minimum distance between unique genes in the pathway, if it is 0, average dispersion is performed, the default value is `0`.
        - `--forward`:Whether it is positive or not, he indicates whether the gene pairs being screened meet the criterion `gene_relationship_file`, if it is True, then it meets the criterion, and vice versa, it is taking the opposite, screening out gene pairs that don't meet the `gene_relationship_file`, which is used to make prediction discovery. The default value is `True`.
- `interblock`

    - **Function**: To compare differences between groups.
    - **Function parameters**
        - `--input_A`:Target file A, the file in which the difference comparison is performed, in the format `.txt`.
        - `--input_B`:Target file B, the file in which the difference comparison is performed, in the format `.txt`.
        - `--pathway_file`: The target pathway, the pathway to be visualised, is formatted with the first column as the pathway name and the rest of the columns as the genes in the pathway, and the file is an excel spreadsheet.
        - `--output_pdf`: The address of the image output, in a format ending in `.pdf`.
        - `--output_file`: The address of the file output, in this case a folder.
        - `--center_distance_A`:Proportional distance of part A from the centre of the canvas, with a default value of `0.5`.
        - `--center_distance_B`:Proportional distance of part B from the centre of the canvas, with a default value of `0.5`.
        - `--center_distance_intersecting`:Proportional distance of the common parts from the centre of the canvas, default `0.5`.
        - `--radius_A`：The radius scale for part A, with a default value of `0.3`.
        - `--radius_B`：The radius scale for part B, with a default value of `0.3`.
        - `--radius_intersecting`：The radius ratio of the common parts, with a default value of `0.5`.
        - `--figure_size`: Image size, the image size can be adjusted according to the number of genes, the default value is `15`.
        - `--fontsize_intersecting`:The font size of the public section, defaults to `12`.
        - `--fontsize_others`:The font size for the rest of the text, which defaults to `12`.

---------

#### code example

```bash
# Drawing intra-group comparisons
python pathway.py intra --pathway_file "./utils/pathway/pathways.xlsx" --input_txt "./DATA/example/cancer.txt" --output_file './DATA/pathway_result' 

# Drawing intergroup comparisons
python pathway.py interblock --input_A './DATA/example/A.txt' --input_B './DATA/example/B.txt' --pathway_file "./utils/pathway/pathways.xlsx" --output_pdf './DATA/pathway_result/A_B.pdf' --output_file './DATA/pathway_result' --figure_size 30
 
``` 

## Project structure
```
PyGeoNet/
├── README.md                  # Project Documentation
├── requirements.txt           # List of dependency packages
├── search.py                  # Data search module (filtering GSE numbers based on keywords)
├── download.py                # Data download module (download raw data based on GSE number)
├── normalize.py               # Data normalisation module (processing of raw data, generation of gene expression matrices)
├── analyse.py                 # Variance analysis module (generating visualisations such as volcano/heat maps)
├── standard.py                # Graph model input preprocessing module (calculating gene similarity, PCA downscaling, etc.)
├── model.py                   # Graph Neural Network Training and Prediction Module (supports GAT/GCN and other models)
├── pathway.py                 # Pathway difference visualisation module (intra/intergroup pathway difference mapping)
├── utils/                     # Tool Functions Catalogue
    ├── normalize/
    │   └── tool.py            # Tool functions related to data normalisation
    ├── model/
    │   ├── class_tool.py      # Classes of tools related to graph model training (e.g. GAEModel, GATModel, etc.)
    │   └── edge_tool.py       # Edge processing related tool functions
    └── pathway/
        └── pathways.xlsx      # Access Information Form (example)


```
