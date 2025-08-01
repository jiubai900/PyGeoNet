import argparse
import os
import pandas as pd
import networkx as nx
import numpy as np
from matplotlib.patches import Ellipse, Circle
import matplotlib.pyplot as plt
import matplotlib as mpl
from adjustText import adjust_text

mpl.rcParams['pdf.fonttype'] = 42


def read_pathway_file(file_path):
    """Read access file, support XLSX and GMT two formats"""
    if file_path.endswith('.xlsx'):
        pathways_df = pd.read_excel(file_path)
        pathway_genes = {}
        for _, row in pathways_df.iterrows():
            pathway_name = row.iloc[0]
            genes = row.iloc[1:].dropna().values
            pathway_genes[pathway_name] = set(genes)
        return pathway_genes
    elif file_path.endswith('.gmt'):
        pathway_genes = {}
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                pathway_name = parts[0]
                genes = set(parts[2:])  # The first two columns in the GMT format are ID and description
                pathway_genes[pathway_name] = genes
        return pathway_genes
    else:
        raise ValueError("Unsupported file format. Only .xlsx and .gmt are supported.")


def intra(pathway_file, input_txt, output_file, gene_relationship_file, max_pathways,
          figure_size, max_pathway_radius, min_pathway_radius, gene_radius, forward):
    print(f'forward:{forward}')

    # Modify save_path to the folder path
    save_folder = output_file
    os.makedirs(save_folder, exist_ok=True)

    # Read pathway data
    pathway_genes = read_pathway_file(pathway_file)

    unique_genes_dict = {}

    # Read gene relationship data
    gene_relationship_df = pd.read_csv(gene_relationship_file, sep='\t', header=None, dtype={0: str, 1: str},
                                       low_memory=False)
    gene_relationships = set(tuple(sorted(row)) for row in gene_relationship_df[[0, 1]].values)  # 添加sorted

    # Reading model result data
    model_results_df = pd.read_csv(input_txt, sep='\t', header=None, dtype={0: str, 1: str}, low_memory=False)
    model_relationships = set(tuple(sorted(row)) for row in model_results_df.values)  # 添加sorted

    # Screening effective edges
    if forward:
        valid_edges = model_relationships.intersection(gene_relationships)
    else:
        valid_edges = model_relationships.difference(gene_relationships)

    # Extract file names (without paths and extensions)
    file_name = os.path.splitext(os.path.basename(input_txt))[0]

    shared_edges = set()
    # Grouping based on access
    pathway_edges = {pathway: [] for pathway in pathway_genes}
    pathway_genes_in_edges = {pathway: set() for pathway in pathway_genes}
    for edge in valid_edges:  # 使用 filtered_valid_edges 替换 valid_edges
        gene1, gene2 = edge
        containing_pathways = []
        for pathway, genes in pathway_genes.items():
            if gene1 in genes and gene2 in genes:
                pathway_edges[pathway].append(edge)
                pathway_genes_in_edges[pathway].update(edge)
                containing_pathways.append(pathway)
        if len(containing_pathways) > 1:
            shared_edges.add(edge)

    # Calculate the number of gene pairs for each pathway and select the top max_pathways

    pathway_edge_counts = {pathway: len(edges) for pathway, edges in pathway_edges.items()}
    print(pathway_edge_counts)
    top_pathways = sorted(pathway_edge_counts.items(), key=lambda x: x[1], reverse=True)[:max_pathways]
    top_pathways = [pathway for pathway, _ in top_pathways]

    # Filtering out pathways without unique genes
    ordered_pathways = []
    for pathway in top_pathways:
        unique_genes = pathway_genes_in_edges[pathway].copy()
        for other_pathway in top_pathways:
            if other_pathway != pathway:
                unique_genes -= pathway_genes_in_edges[other_pathway]
        if len(unique_genes) > 0:
            ordered_pathways.append(pathway)

    if len(ordered_pathways) == 0:
        print("No pathway containing a unique gene was found。")
        return

    # reconstruct pathway_genes and valid_edges
    pathway_genes = {pathway: pathway_genes[pathway] for pathway in ordered_pathways}
    valid_edges = set()
    for pathway in ordered_pathways:
        valid_edges.update(pathway_edges[pathway])

    all_pathways_edges_path = os.path.join(save_folder, f"{file_name}_all_pathways_edges.txt")
    with open(all_pathways_edges_path, 'w') as f:
        f.write("gene1\tgene2\tpathway\n")  # 添加标题行
        for pathway, edges in pathway_edges.items():
            for gene1, gene2 in edges:
                f.write(f"{gene1}\t{gene2}\t{pathway}\n")

    # reconstruct pathway_edges and pathway_genes_in_edges
    pathway_edges = {pathway: [] for pathway in pathway_genes}
    pathway_genes_in_edges = {pathway: set() for pathway in pathway_genes}
    for edge in valid_edges:
        gene1, gene2 = edge
        for pathway, genes in pathway_genes.items():
            if gene1 in genes and gene2 in genes:
                pathway_edges[pathway].append(edge)
                pathway_genes_in_edges[pathway].update(edge)

    # Calculate the number of unique genes per pathway
    unique_gene_counts = {}
    for pathway in ordered_pathways:
        unique_genes = pathway_genes_in_edges[pathway].copy()
        for other_pathway in ordered_pathways:
            if other_pathway != pathway:
                unique_genes -= pathway_genes_in_edges[other_pathway]
        unique_genes_dict[pathway] = unique_genes
        unique_gene_counts[pathway] = len(unique_genes)
    # Calculation of genetic overlap between pathways
    overlap_matrix = np.zeros((len(ordered_pathways), len(ordered_pathways)))
    pathway_names = list(ordered_pathways)
    pathway_indices = {name: idx for idx, name in enumerate(pathway_names)}
    for i, pathway1 in enumerate(ordered_pathways):
        for j, pathway2 in enumerate(ordered_pathways):
            if i == j:
                overlap_matrix[i][j] = 0
            else:
                overlap_genes = pathway_genes_in_edges[pathway1].intersection(pathway_genes_in_edges[pathway2])
                overlap_matrix[i][j] = len(overlap_genes)

    def determine_pathway_order(filtered_pathways, overlap_matrix):
        # Number of access routes acquired
        num_pathways = len(filtered_pathways)

        # Create a list to hold the order of the sorted pathways
        sorted_pathways = []

        # Create a dictionary that stores the sum of the number of overlapping genes for each pathway and its neighbouring pathways
        overlap_sums = {}
        for i, pathway in enumerate(filtered_pathways):
            # Get the number of overlapping genes in the current row (i.e., the number of overlaps of the current pathway with other pathways)
            overlap_scores = overlap_matrix[i]

            # Take the sum of the two largest numbers
            largest_two = sorted(overlap_scores, reverse=True)[:2]
            overlap_sum = sum(largest_two)  # Calculate the sum of the two largest numbers
            overlap_sums[pathway] = overlap_sum

        # The pathway with the largest number of overlapping genes and the largest number of overlapping genes was chosen as the starting point
        start_pathway = max(overlap_sums, key=overlap_sums.get)
        sorted_pathways.append(start_pathway)

        # Recording of sorted pathways
        remaining_pathways = set(filtered_pathways) - {start_pathway}

        # Identify the two neighbours of the first access road
        start_idx = filtered_pathways.index(start_pathway)
        overlap_scores = overlap_matrix[start_idx]

        # Selection of the two neighbours with the largest number of overlapping genes
        neighbors = [(filtered_pathways[i], overlap_scores[i]) for i in range(num_pathways) if
                     filtered_pathways[i] in remaining_pathways]
        neighbors.sort(key=lambda x: x[1], reverse=True)

        if len(neighbors) >= 2:
            left_neighbor, right_neighbor = neighbors[0][0], neighbors[1][0]
        else:
            # Handling neighbors with insufficient elements
            left_neighbor = neighbors[0][0] if neighbors else None
            right_neighbor = None

        # After determining the left and right neighbours, continue to select the neighbours of the remaining unordered pathways
        while remaining_pathways:
            # Iterate through the header and footer paths in the current list
            first_pathway = sorted_pathways[0]
            last_pathway = sorted_pathways[-1]

            # Obtaining the number of overlapping genes with unordered pathways
            first_idx = filtered_pathways.index(first_pathway)
            last_idx = filtered_pathways.index(last_pathway)

            # Get the number of overlapping genes with the remaining pathways
            first_overlap = overlap_matrix[first_idx]
            last_overlap = overlap_matrix[last_idx]

            # Screening for unranked pathways with the largest number of overlapping genes with the head and tail
            possible_neighbors = []
            for i in range(num_pathways):
                if filtered_pathways[i] in remaining_pathways:
                    # Add the sum of the number of overlaps with the current header and footer pathways
                    overlap_sum = first_overlap[i] + last_overlap[i]
                    possible_neighbors.append((filtered_pathways[i], overlap_sum))

            # If there are pathways with overlapping gene counts, select the pathway with the largest
            if possible_neighbors:
                possible_neighbors.sort(key=lambda x: x[1], reverse=True)
                best_neighbor = possible_neighbors[0][0]
            else:
                # If there are no overlapping genes, insert a random
                best_neighbor = remaining_pathways.pop()

            # Insert the most appropriate pathway into the head or tail of the list
            if first_overlap[filtered_pathways.index(best_neighbor)] >= last_overlap[
                filtered_pathways.index(best_neighbor)]:
                sorted_pathways.insert(0, best_neighbor)
            else:
                sorted_pathways.append(best_neighbor)

            remaining_pathways.remove(best_neighbor)

        return sorted_pathways

    # Call determine_pathway_order to sort the pathway
    ordered_pathways = determine_pathway_order(ordered_pathways, overlap_matrix)
    max_pathway_genes = set().union(*[pathway_genes[pathway] for pathway in ordered_pathways])
    filtered_valid_edges = {edge for edge in valid_edges if
                            edge[0] in max_pathway_genes and edge[1] in max_pathway_genes}

    # Grouping based on access
    pathway_edges = {pathway: [] for pathway in pathway_genes}
    pathway_genes_in_edges = {pathway: set() for pathway in pathway_genes}
    for edge in filtered_valid_edges:  # use filtered_valid_edges interchangeability valid_edges
        gene1, gene2 = edge
        for pathway, genes in pathway_genes.items():
            if gene1 in genes and gene2 in genes:
                pathway_edges[pathway].append(edge)
                pathway_genes_in_edges[pathway].update(edge)

    adjacent_overlap_genes = {}
    for i in range(len(ordered_pathways)):
        pathway1 = ordered_pathways[i]
        pathway2 = ordered_pathways[(i + 1) % len(ordered_pathways)]  # When the loop reaches the last pathway, the next pathway is the first pathway
        overlap_genes = pathway_genes_in_edges[pathway1].intersection(pathway_genes_in_edges[pathway2])

        # Further filtering to ensure that these genes are not in other pathways
        for other_pathway in ordered_pathways:
            if other_pathway != pathway1 and other_pathway != pathway2:
                overlap_genes -= pathway_genes_in_edges[other_pathway]

        adjacent_overlap_genes[(pathway1, pathway2)] = overlap_genes
    # Determine the size and location of the area of the access road
    total_area = 100  # Suppose the total area of the canvas is 100 units
    max_unique_genes = max(unique_gene_counts.values())
    min_radius = min_pathway_radius  # Minimum radius, halved.
    max_radius = max_pathway_radius  # Maximum radius

    # Calculate the location of the access area
    positions = {}
    angles = np.linspace(0, 2 * np.pi, len(ordered_pathways), endpoint=False)
    center_x, center_y = 50, 50  # Canvas Centre
    distance_from_center = 35  # Reduce the distance of the access circle from the centre of the canvas.

    for angle, pathway in zip(angles, ordered_pathways):
        num_unique_genes = unique_gene_counts[pathway]
        radius = min_radius + (max_radius - min_radius) * (num_unique_genes / max_unique_genes)  # Adjustment of radius according to the number of unique genes
        x = center_x + distance_from_center * np.cos(angle)
        y = center_y + distance_from_center * np.sin(angle)
        positions[pathway] = (x, y, radius)

    # Setting the position for each gene
    gene_positions = {}
    gene_regions = {}

    # Calculate the number of intermediate genes
    all_genes = set()
    for pathway in ordered_pathways:
        all_genes.update(pathway_genes_in_edges[pathway])

    central_region_genes = all_genes.copy()
    for pathway in ordered_pathways:
        central_region_genes -= unique_genes_dict[pathway]
        for other_pathway in ordered_pathways:
            if other_pathway != pathway:
                central_region_genes -= adjacent_overlap_genes.get((pathway, other_pathway), set())

    # Calculate the number of intermediate genes
    num_overlapping_genes = len(central_region_genes)

    # Calculate the size of the circular region of the intermediate gene
    if num_overlapping_genes > 0:
        radius = min(max(len(central_region_genes) / 50, 10), 35 - max_pathway_radius)
        for idx, gene in enumerate(central_region_genes):
            theta = np.random.uniform(0, 2 * np.pi)
            d = np.random.uniform(0, radius)
            gene_x = center_x + d * np.cos(theta)
            gene_y = center_y + d * np.sin(theta)
            gene_positions[gene] = (gene_x, gene_y)
            if gene not in gene_regions:
                gene_regions[gene] = []
            gene_regions[gene].extend(
                [pathway for pathway in ordered_pathways if gene in pathway_genes_in_edges[pathway]])

    # Dealing with overlapping genes
    for (pathway1, pathway2), overlap_genes in adjacent_overlap_genes.items():
        if overlap_genes:
            x1, y1, r1 = positions[pathway1]
            x2, y2, r2 = positions[pathway2]
            midpoint_x = (x1 + x2) / 2
            midpoint_y = (y1 + y2) / 2
            long_axis = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            short_axis = long_axis / 5  # Short axle one fifth of long axle

            # 计算椭圆的角度
            angle_rad = np.arctan2(y2 - y1, x2 - x1)
            cos_angle = np.cos(angle_rad)
            sin_angle = np.sin(angle_rad)

            for gene in overlap_genes:
                # Randomly select an angle theta between [0, 2*pi)
                theta = np.random.uniform(0, 2 * np.pi)
                # Randomly select a radius factor r between [0, 1].
                r = np.sqrt(np.random.uniform(0, 1))  # Ensure even distribution

                # Calculate the points in the ellipse
                x_ellipse = r * long_axis / 2 * np.cos(theta)
                y_ellipse = r * short_axis / 2 * np.sin(theta)

                # Rotate and translate the point inside the ellipse to the centre point
                gene_x = midpoint_x + x_ellipse * cos_angle - y_ellipse * sin_angle
                gene_y = midpoint_y + x_ellipse * sin_angle + y_ellipse * cos_angle

                gene_positions[gene] = (gene_x, gene_y)
                if gene not in gene_regions:
                    gene_regions[gene] = []
                gene_regions[gene].extend([pathway1, pathway2])

    def calculate_layer_radii(radius, min_radius, decay_factor=0.7):
        """
        Calculate the radius of each layer.

        :param radius: Maximum radius
        :param min_radius: Minimum radius
        :param decay_factor: Radius attenuation factor per layer
        :return: radius list
        """
        layers = []
        current_radius = radius
        while current_radius > min_radius:
            layers.append(current_radius)
            current_radius *= decay_factor
        if current_radius >= min_radius:
            layers.append(min_radius)
        return layers

    # Calculate the radius of each layer
    # Increase the minimum spacing between genes

    # Setting positions for genes in each pathway
    for pathway, genes in pathway_genes.items():
        x, y, r = positions.get(pathway, (0, 0, 0))
        if r == 0:  # Make sure the radius is not zero
            r = min_radius  # Setting the default minimum radius
        genes_in_pathway = pathway_genes_in_edges[pathway]
        unique_genes = genes_in_pathway.copy()
        for other_pathway in ordered_pathways:
            if other_pathway != pathway:
                unique_genes -= pathway_genes_in_edges[other_pathway]

        # Calculate the radius of each layer
        layer_radii = calculate_layer_radii(r, min_radius)

        # Place unique genes first
        num_unique_genes = len(unique_genes)
        if num_unique_genes > 0:
            total_genes_placed = 0
            unique_genes_list = list(unique_genes)  # Converting a collection to a list
            for layer_idx, layer_radius in enumerate(layer_radii):
                num_genes_in_layer = int(np.ceil(num_unique_genes * (layer_radius / r) ** 2))
                if num_genes_in_layer == 0:
                    continue
                angles = np.linspace(0, 2 * np.pi, num_genes_in_layer, endpoint=False)
                for idx, gene in enumerate(
                        unique_genes_list[total_genes_placed:total_genes_placed + num_genes_in_layer]):
                    angle = angles[idx]
                    gene_x = x + layer_radius * np.cos(angle)
                    gene_y = y + layer_radius * np.sin(angle)

                    # Check the distance between neighbouring genes
                    for existing_gene, (existing_x, existing_y) in gene_positions.items():
                        distance = np.sqrt((gene_x - existing_x) ** 2 + (gene_y - existing_y) ** 2)
                        if distance < gene_radius * 2:
                            # If the distance is less than the minimum interval, recalculate the position
                            angle += np.pi / num_genes_in_layer
                            gene_x = x + layer_radius * np.cos(angle)
                            gene_y = y + layer_radius * np.sin(angle)

                    gene_positions[gene] = (gene_x, gene_y)
                    if gene not in gene_regions:
                        gene_regions[gene] = []
                    gene_regions[gene].append(pathway)
                total_genes_placed += num_genes_in_layer

            # If there are any genes left, put them in the last layer
            if total_genes_placed < num_unique_genes:
                remaining_genes = unique_genes_list[total_genes_placed:]
                angles = np.linspace(0, 2 * np.pi, len(remaining_genes), endpoint=False)
                for idx, gene in enumerate(remaining_genes):
                    angle = angles[idx]
                    gene_x = x + min_radius * np.cos(angle)
                    gene_y = y + min_radius * np.sin(angle)

                    # Check the distance between neighbouring genes
                    for existing_gene, (existing_x, existing_y) in gene_positions.items():
                        distance = np.sqrt((gene_x - existing_x) ** 2 + (gene_y - existing_y) ** 2)
                        if distance < gene_radius * 2:
                            # If the distance is less than the minimum interval, recalculate the position
                            angle += np.pi / len(remaining_genes)
                            gene_x = x + min_radius * np.cos(angle)
                            gene_y = y + min_radius * np.sin(angle)

                    gene_positions[gene] = (gene_x, gene_y)
                    if gene not in gene_regions:
                        gene_regions[gene] = []
                    gene_regions[gene].append(pathway)

    # Creating a Network
    G = nx.Graph()
    G.add_edges_from(valid_edges)

    # Dynamically resize gene nodes and font sizes
    total_genes = len(all_genes)
    if total_genes > 500:
        node_size = 10
        font_size = 8
    elif total_genes > 200:
        node_size = 20
        font_size = 10
    elif total_genes > 100:
        node_size = 30
        font_size = 12
    elif total_genes > 50:
        node_size = 40
        font_size = 16
    else:
        node_size = 70
        font_size = 20

    # Assign a unique colour to each pathway
    cmap = plt.get_cmap('tab10')
    pathway_colors = {pathway: cmap(i % cmap.N) for i, pathway in enumerate(ordered_pathways)}

    # Determine the order in which the pathways are arranged
    # Update pathway locations to ensure correct overlap areas between neighbouring pathways
    angles = np.linspace(0, 2 * np.pi, len(ordered_pathways), endpoint=False)
    for angle, pathway in zip(angles, ordered_pathways):
        num_unique_genes = unique_gene_counts[pathway]
        radius = min_radius + (max_radius - min_radius) * (num_unique_genes / max_unique_genes)  # Adjustment of radius according to the number of unique genes
        x = center_x + distance_from_center * np.cos(angle)
        y = center_y + distance_from_center * np.sin(angle)
        positions[pathway] = (x, y, radius)

    # Debug Printing
    print("Filtered Pathways:", ordered_pathways)
    print("Ordered Pathways:", ordered_pathways)
    print("Pathway Colors:", pathway_colors)

    # 绘图函数
    def plot_pathway_graph(positions, gene_positions, gene_regions, ordered_pathways, node_size, font_size,
                           pathway_colors, save_path, figure_size):
        fig, ax = plt.subplots(figsize=(figure_size, figure_size))  # Resizing the canvas

        # Drawing the background colour of the access road
        for pathway, (x, y, r) in positions.items():
            circle = Circle((x, y), r, color=pathway_colors[pathway], alpha=0.2)
            ax.add_patch(circle)

        # Draws background colour of overlapping areas, only overlapping areas between adjacent pathways are drawn
        for (pathway1, pathway2), overlap_genes in adjacent_overlap_genes.items():
            if len(overlap_genes) > 0:  # Check that the number of overlapping genes is greater than 0
                x1, y1, r1 = positions[pathway1]
                x2, y2, r2 = positions[pathway2]
                midpoint_x = (x1 + x2) / 2
                midpoint_y = (y1 + y2) / 2
                long_axis = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                short_axis = long_axis / 5  # Short axle one fifth of long axle
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))  # 计算角度
                ellipse = Ellipse((midpoint_x, midpoint_y), long_axis, short_axis, angle=angle, color='gray', alpha=0.2)
                ax.add_patch(ellipse)

        # draw a border (i.e. plot the edges of a triangle)
        for pathway, edges in pathway_edges.items():
            for edge in edges:
                gene1, gene2 = edge
                # Ensure that both gene1 and gene2 have a place in gene_positions
                if gene1 in gene_positions and gene2 in gene_positions:
                    x1, y1 = gene_positions[gene1]
                    x2, y2 = gene_positions[gene2]
                    # Make sure the pathway is in ordered_pathways
                    if pathway in ordered_pathways:
                        # Calculate the points from the centre of the node to the edge of the node
                        dx1, dy1 = x2 - x1, y2 - y1
                        length1 = np.sqrt(dx1 ** 2 + dy1 ** 2)
                        if length1 > 0:
                            ex1, ey1 = x1 + dx1 * gene_radius / length1, y1 + dy1 * gene_radius / length1
                        else:
                            ex1, ey1 = x1, y1

                        dx2, dy2 = x1 - x2, y1 - y2
                        length2 = np.sqrt(dx2 ** 2 + dy2 ** 2)
                        if length2 > 0:
                            ex2, ey2 = x2 + dx2 * gene_radius / length2, y2 + dy2 * gene_radius / length2
                        else:
                            ex2, ey2 = x2, y2

                        plt.plot([ex1, ex2], [ey1, ey2], color=pathway_colors[pathway], alpha=0.3, linewidth=1)

        # Drawing shared edges
        for edge in shared_edges:
            gene1, gene2 = edge
            # Ensure that both gene1 and gene2 have a place in gene_positions
            if gene1 in gene_positions and gene2 in gene_positions:
                x1, y1 = gene_positions[gene1]
                x2, y2 = gene_positions[gene2]
                # Calculate the points from the centre of the node to the edge of the node
                dx1, dy1 = x2 - x1, y2 - y1
                length1 = np.sqrt(dx1 ** 2 + dy1 ** 2)
                if length1 > 0:
                    ex1, ey1 = x1 + dx1 * gene_radius / length1, y1 + dy1 * gene_radius / length1
                else:
                    ex1, ey1 = x1, y1

                dx2, dy2 = x1 - x2, y1 - y2
                length2 = np.sqrt(dx2 ** 2 + dy2 ** 2)
                if length2 > 0:
                    ex2, ey2 = x2 + dx2 * gene_radius / length2, y2 + dy2 * gene_radius / length2
                else:
                    ex2, ey2 = x2, y2

                plt.plot([ex1, ex2], [ey1, ey2], color='gray', alpha=0.3, linewidth=1)

        # Mapping gene nodes
        texts = []
        for gene, (x, y) in gene_positions.items():
            regions = gene_regions[gene]
            if len(regions) == 1:
                plt.scatter(x, y, s=node_size, color='skyblue', edgecolors='black', alpha=1)
            elif len(regions) == 2:
                plt.scatter(x, y, s=node_size, color='orange', edgecolors='black', alpha=1)
            else:
                plt.scatter(x, y, s=node_size, color='red', edgecolors='black', alpha=1)

            # Adjustment of gene name display position
            offset_x = 0
            offset_y = 0
            text = plt.text(x + offset_x, y + offset_y, gene, fontsize=font_size, ha='center', va='center',
                            color='black')
            texts.append(text)

        # Adding a pathway name
        for pathway, (x, y, r) in positions.items():
            text = plt.text(x, y, pathway, fontsize=font_size + 2, ha='center', va='center', color='red')
            texts.append(text)

        # Remove the axes
        plt.axis('off')

        # Adjust layout to avoid inexplicable stuff in the bottom left corner
        ax.set_xlim(center_x - 60, center_x + 60)  # Adding borders
        ax.set_ylim(center_y - 60, center_y + 60)  # Adding borders
        plt.gca().set_aspect('equal', adjustable='box')
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color='black', lw=1),
                    textprops=dict(fontfamily='Arial', fontsize=font_size))
        plt.tight_layout()
        plt.savefig(save_path, format='pdf')

    # Save side as txt file
    edges_file_path = os.path.join(save_folder, f"{file_name}_edge.txt")
    with open(edges_file_path, 'w') as edges_file:
        for edge in valid_edges:
            edges_file.write(f"{edge[0]}\t{edge[1]}\n")

    # Save gene as txt file
    genes_file_path = os.path.join(save_folder, f"{file_name}_gene.txt")
    with open(genes_file_path, 'w') as genes_file:
        for gene in all_genes:
            genes_file.write(f"{gene}\n")

    # Modify image save path and format
    image_save_path = os.path.join(save_folder, f"{file_name}.pdf")

    # Use the new image save path in plot_pathway_graph function calls
    plot_pathway_graph(positions, gene_positions, gene_regions, ordered_pathways, node_size, font_size,
                       pathway_colors, image_save_path, figure_size)


def interblock(input_A, input_B, file_pathways, output_pdf, output_file,
               center_distance_A_B, center_distance_B_A,
               center_distance_A_intersect_B, radius_A_B, radius_B_A,
               radius_A_intersect_B, figure_size, fontsize_A_intersect_B, fontsize_others):
    """
    Complete processing flow: read file, filter gene edges, calculate intersection and difference, draw circles and edges, and label the number of genes
    parametric：
    - file_A: A.txt file path
    - file_B: B.txt file path
    - file_pathways: Access Information Excel File Path
    - save_path: Path to save the graphic
    - save_folder_path: Path to the folder where the filter results are saved
    - center_distance_A_B: Distance between the centre of the A-B area and the centre of the canvas
    - center_distance_B_A: Distance between the centre of the B-A area and the centre of the canvas
    - center_distance_A_intersect_B: Distance between the centre of the area where A intersects B and the centre of the canvas
    - radius_A_B: Radius of area A-B
    - radius_B_A: Radius of the B-A region
    - radius_A_intersect_B: A intersects the radius of region B
    - figure_size: Size of graphics
    - fontsize_A_intersect_B: Font size in the A to B region
    - fontsize_others: Font size in other areas
    """
    if not os.path.exists(output_file):
        os.makedirs(output_file)

    file_A_name = os.path.basename(input_A)[:-4]
    file_B_name = os.path.basename(input_B)[:-4]
    # Read the gene side files of A and B
    edges_A = pd.read_csv(input_A, sep="\t", header=0, names=["gene1", "gene2"])
    edges_B = pd.read_csv(input_B, sep="\t", header=0, names=["gene1", "gene2"])

    # Read pathway information
    pathways_df = pd.read_excel(file_pathways, header=0, index_col=0)
    pathways = {pathway_name: set(genes.dropna()) for pathway_name, genes in pathways_df.iterrows()}

    # Screening the gene edges of A and B to ensure that genes from each edge occur in the same pathway
    filtered_edges_A = []
    filtered_edges_B = []

    for _, row in edges_A.iterrows():
        gene1, gene2 = row["gene1"], row["gene2"]
        for pathway_genes in pathways.values():
            if gene1 in pathway_genes and gene2 in pathway_genes:
                filtered_edges_A.append((gene1, gene2))
                break

    for _, row in edges_B.iterrows():
        gene1, gene2 = row["gene1"], row["gene2"]
        for pathway_genes in pathways.values():
            if gene1 in pathway_genes and gene2 in pathway_genes:
                filtered_edges_B.append((gene1, gene2))
                break

    # Convert to a collection
    set_A = set(filtered_edges_A)
    set_B = set(filtered_edges_B)

    # Extract all genes
    genes_A = set()
    genes_B = set()
    for gene1, gene2 in set_A:
        genes_A.add(gene1)
        genes_A.add(gene2)
    for gene1, gene2 in set_B:
        genes_B.add(gene1)
        genes_B.add(gene2)

    # Calculate A-B, B-A and A to B.
    genes_A_minus_B = genes_A - genes_B
    genes_B_minus_A = genes_B - genes_A
    genes_A_intersection_B = genes_A & genes_B

    # Compute the intersection and difference of edges
    set_A_minus_B = set_A - set_B
    set_B_minus_A = set_B - set_A
    set_A_intersection_B = set_A & set_B

    # Extract all genes and the set to which they belong
    gene_positions = {}
    for gene in genes_A_minus_B:
        gene_positions[gene] = {'set': 'A-B', 'points': []}
    for gene in genes_B_minus_A:
        gene_positions[gene] = {'set': 'B-A', 'points': []}
    for gene in genes_A_intersection_B:
        gene_positions[gene] = {'set': 'A交B', 'points': []}

    centers = {
        'A-B': (center_distance_A_B * np.cos(np.radians(225)), center_distance_A_B * np.sin(np.radians(225))),
        'B-A': (center_distance_B_A * np.cos(np.radians(315)), center_distance_B_A * np.sin(np.radians(315))),
        'A交B': (
            center_distance_A_intersect_B * np.cos(np.radians(90)),
            center_distance_A_intersect_B * np.sin(np.radians(90)))
    }
    radii = {
        'A-B': radius_A_B,
        'B-A': radius_B_A,
        'A交B': radius_A_intersect_B
    }

    # Creating a drawing
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))

    # Draw circular regions corresponding to A-B, B-A and A intersecting B.
    def plot_circle(center, radius, color, alpha=1):
        circle = Circle(center, radius, color=color, alpha=alpha)
        ax.add_patch(circle)

    plot_circle(center=centers['A-B'], radius=radii['A-B'], color='darkred', alpha=0.2)  # Region A-B
    plot_circle(center=centers['B-A'], radius=radii['B-A'], color='darkgreen', alpha=0.2)  # Region B-A
    plot_circle(center=centers['A交B'], radius=radii['A交B'], color='gray', alpha=0.2)  # Region A to B

    # Randomly generate points A-B, B-A and A intersecting B, corresponding to different sets respectively
    def random_points_in_circle(center, radius, n_points):
        points = []
        for _ in range(n_points):
            theta = np.random.uniform(0, 2 * np.pi)
            r = radius * np.sqrt(np.random.uniform(0, 1))
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            points.append((x, y))
        return points

    # Assign points to each gene
    for gene, info in gene_positions.items():
        if info['set'] == 'A-B':
            info['points'].extend(random_points_in_circle(centers['A-B'], radii['A-B'], 1))
        elif info['set'] == 'B-A':
            info['points'].extend(random_points_in_circle(centers['B-A'], radii['B-A'], 1))
        elif info['set'] == 'A交B':
            info['points'].extend(random_points_in_circle(centers['A交B'], radii['A交B'], 1))

    # draw borders
    def draw_edges(edges, color, alpha):
        for gene1, gene2 in edges:
            if gene1 in gene_positions and gene2 in gene_positions:
                x1, y1 = gene_positions[gene1]['points'][0]
                x2, y2 = gene_positions[gene2]['points'][0]
                ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha)

    # Drawing the edges of the A-B area
    draw_edges(set_A_minus_B, color='darkred', alpha=0.15)  # Lighten the colour of the edges
    # Drawing the edges of the B-A region
    draw_edges(set_B_minus_A, color='darkgreen', alpha=0.15)  # Lighten the colour of the edges
    # Drawing A intersects the edge of region B
    draw_edges(set_A_intersection_B, color='black', alpha=0.15)  # Lighten the colour of the edges

    # Display gene name
    texts = []
    for gene, info in gene_positions.items():
        x, y = info['points'][0]
        color = 'orange'  # Set all node colours to orange
        ax.scatter(x, y, color=color, edgecolors='black', s=100)  # Adding a black edge
        if info['set'] == 'A交B':
            fontsize = fontsize_A_intersect_B
        else:
            fontsize = fontsize_others
        text = ax.text(x, y, gene, fontsize=fontsize, ha='center', va='center', color='black')  # 显示基因名称
        texts.append(text)

    # Use the adjustText library to automatically reposition text to avoid overlapping.
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="->", color='black', lw=0.5))

    # Number of genes labelled
    # The number of genes is labelled below the A-B region.
    ax.text(centers['A-B'][0], centers['A-B'][1] - radii['A-B'] - 0.05,
            f'{file_A_name}:{len(genes_A_minus_B)}', fontsize=30, ha='center', va='top', color='darkred')
    # The number of genes is labelled below the B-A region.
    ax.text(centers['B-A'][0], centers['B-A'][1] - radii['B-A'] - 0.05,
            f'{file_B_name}:{len(genes_B_minus_A)}', fontsize=30, ha='center', va='top',
            color='darkgreen')
    # Number of genes labelled above the B region of the A-crossing.
    ax.text(centers['A交B'][0], centers['A交B'][1] + radii['A交B'] + 0.05,
            f'intersection:{len(genes_A_intersection_B)}', fontsize=30, ha='center', va='bottom', color='gray')

    # Setting Graphic Properties
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    plt.axis('off')  # Close Axis
    plt.savefig(output_pdf)

    # Create save folder (if it does not exist)
    if not os.path.exists(output_file):
        os.makedirs(output_file)

    set_out_A = {(gene1, gene2) for gene1, gene2 in set_A if gene1 in genes_A_minus_B and gene2 in genes_A_minus_B}
    set_out_B = {(gene1, gene2) for gene1, gene2 in set_B if gene1 in genes_B_minus_A and gene2 in genes_B_minus_A}

    # Save the results to a different file
    save_file_A = os.path.join(output_file, f"{file_A_name}_unique_edges.txt")
    save_file_B = os.path.join(output_file, f"{file_B_name}_unique_edges.txt")

    with open(save_file_A, 'w') as f:
        f.write("gene1\tgene2\n")
        for gene1, gene2 in set_out_A:
            f.write(f"{gene1}\t{gene2}\n")

    with open(save_file_B, 'w') as f:
        f.write("gene1\tgene2\n")
        for gene1, gene2 in set_out_B:
            f.write(f"{gene1}\t{gene2}\n")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    # Create an argument parser for command-line interaction
    parser = argparse.ArgumentParser(description="Process and visualize data for intra and interblock analysis.")

    # Required arguments
    parser.add_argument("mode", choices=["intra", "interblock"],
                        help="Choose operation mode: 'intra' or 'interblock'")

    # Common arguments for both intra and interblock
    parser.add_argument("--pathway_file", type=str, required=True, help="Pathway file for analysis.")
    parser.add_argument("--output_file", type=str, required=True, help="Output file path for results.")

    # Specific arguments for intra analysis
    parser.add_argument("--input_txt", type=str, help="Input text file for intra analysis.")
    parser.add_argument("--gene_relationship_file", type=str, default='utils/pathway/contrast_gene.txt',
                        help="Gene relationship file, default 'contrast_gene.txt'")
    parser.add_argument("--max_pathways", type=float, default=5, help="Maximum number of pathways, default is 5.")
    parser.add_argument("--figure_size", type=float, default=15, help="Figure size, default is 15.")
    parser.add_argument("--max_pathway_radius", type=float, default=15, help="Maximum pathway radius, default is 15.")
    parser.add_argument("--min_pathway_radius", type=float, default=2.5, help="Minimum pathway radius, default is 2.5.")
    parser.add_argument("--gene_radius", type=float, default=0, help="Gene radius, default is 0.")
    parser.add_argument("--forward", type=str2bool, default=True,
                        help="Whether to apply forward mode, default is True.")

    # Specific arguments for interblock analysis
    parser.add_argument("--input_A", type=str, help="Input A file for interblock analysis.")
    parser.add_argument("--input_B", type=str, help="Input B file for interblock analysis.")
    parser.add_argument("--output_pdf", type=str, help="Output PDF file for interblock analysis.")
    parser.add_argument("--center_distance_A", type=float, default=0.5,
                        help="Center distance between A and the midpoint of the canvas, default 0.5.")
    parser.add_argument("--center_distance_B", type=float, default=0.5,
                        help="Center distance between B and the midpoint of the canvas, default 0.5.")
    parser.add_argument("--center_distance_intersecting", type=float, default=0.5,
                        help="Center distance between intersecting part and the midpoint of the canvas, default 0.5.")
    parser.add_argument("--radius_A", type=float, default=0.3, help="Radius for A, default 0.3.")
    parser.add_argument("--radius_B", type=float, default=0.3, help="Radius for B, default 0.3.")
    parser.add_argument("--radius_intersecting", type=float, default=0.5,
                        help="Radius for the intersecting part, default 0.6.")
    parser.add_argument("--fontsize_intersecting", type=float, default=12,
                        help="Font size for intersecting part, default 12.")
    parser.add_argument("--fontsize_others", type=float, default=12, help="Font size for other elements, default 12.")

    args = parser.parse_args()

    # Call the corresponding function based on the mode
    if args.mode == "intra":
        intra(
            pathway_file=args.pathway_file,
            input_txt=args.input_txt,
            output_file=args.output_file,
            gene_relationship_file=args.gene_relationship_file,
            max_pathways=args.max_pathways,
            figure_size=args.figure_size,
            max_pathway_radius=args.max_pathway_radius,
            min_pathway_radius=args.min_pathway_radius,
            gene_radius=args.gene_radius,
            forward=args.forward
        )
    elif args.mode == "interblock":
        interblock(
            input_A=args.input_A,
            input_B=args.input_B,
            file_pathways=args.pathway_file,
            output_pdf=args.output_pdf,
            output_file=args.output_file,
            center_distance_A_B=args.center_distance_A,
            center_distance_B_A=args.center_distance_B,
            center_distance_A_intersect_B=args.center_distance_intersecting,
            radius_A_B=args.radius_A,
            radius_B_A=args.radius_B,
            radius_A_intersect_B=args.radius_intersecting,
            figure_size=args.figure_size,
            fontsize_A_intersect_B=args.fontsize_intersecting,
            fontsize_others=args.fontsize_others
        )

    print(f"Operation '{args.mode}' completed successfully!")


# Entry point for the script
if __name__ == "__main__":
    main()
