import networkx as nx


class GraphHandler:
    def __init__(self, G):
        if isinstance(G, nx.DiGraph):
            raise Exception("Graph should be undirected.")
        if any(degree <= 1 for node, degree in G.degree()):
            print(
                "Graph contains nodes with one or zero edges. These can be removed to reduce the size of the problem."
            )

        self.num_nodes = G.number_of_nodes()
        self.num_edges = G.number_of_edges()

        # ensure graph has labels 0, 1, ..., num_V-1
        G_int = self.__ensure_integer_labels__(G)
        # relabel to make node n-1 the one with maximum degree
        self.G = self.__get_graph_maxdegree_last_node__(G_int)
        # to avoid a deep circuit, we partition the edges into sets which can be executed in parallel
        if not nx.is_isomorphic(G, self.G):
            raise Exception("Something went wrong.")

        self.__minimum_edge_coloring__()

    def __ensure_integer_labels__(self, G):
        # Check if nodes are already labeled as 0 to num_nodes-1
        if set(G.nodes) == set(range(self.num_nodes)):
            return (
                G  # Return the graph unchanged if nodes are already labeled correctly
            )

        # If nodes are not labeled correctly, create a mapping to relabel them
        node_mapping = {node: i for i, node in enumerate(G.nodes)}

        # Create a new graph with relabeled nodes
        H = nx.relabel_nodes(G, node_mapping, copy=True)

        return H

    def __map_colors_to_edges__(self, line_graph_colors, original_graph):
        # Map colors to edges in the original graph G
        color_to_edges = {}

        # Each node in the line graph corresponds to an edge in the original graph G
        for edge_in_line_graph, color in line_graph_colors.items():
            original_edge = edge_in_line_graph  # This is the corresponding edge in G
            if color not in color_to_edges:
                color_to_edges[color] = []
            color_to_edges[color].append(original_edge)

        # Perform consistency check

        # Get the set of all edges in G
        edges_in_G = set(self.G.edges())
        # Get the set of all edges in color_to_edges
        edges_in_coloring = set(
            edge for edges in color_to_edges.values() for edge in edges
        )

        if edges_in_G != edges_in_coloring:
            raise ValueError(
                "The colored edges do not match the edges in the original graph!"
            )

        return color_to_edges

    def __get_graph_maxdegree_last_node__(self, G):
        # Get node of highest degree
        j = sorted(G.degree(), key=lambda x: x[1], reverse=True)[0][0]
        if j == self.num_nodes - 1:
            return G
        else:
            # Create a mapping to swap node j and n-1
            mapping = {j: self.num_nodes - 1, self.num_nodes - 1: j}

            # Relabel the nodes
            H = nx.relabel_nodes(G, mapping, copy=True)

            return H

    def __minimum_edge_coloring__(self, repetitions=100):
        #
        # a graph G
        # returns minimum edge coloring, i.e., a dict containting the edges for each color
        # this can be used to minimize the depth needed to implement diagonal cost Hamiltonians
        # example output
        # { 3: [(0, 1), (2, 7), (3, 9), (5, 6)],
        #   1: [(0, 3), (1, 5), (7, 9)],
        #   2: [(0, 9), (1, 6), (2, 5), (3, 8), (4, 7)],
        #   0: [(1, 4), (2, 9), (3, 7), (5, 8)] }
        #

        # Convert the graph to its line graph
        line_G = nx.line_graph(self.G)

        ncolors = self.num_edges + 1
        for _ in range(repetitions):
            # Apply greedy vertex coloring on the line graph
            line_graph_colors = nx.coloring.greedy_color(
                line_G, strategy="random_sequential"
            )

            # groups of parallel edges
            pe = self.__map_colors_to_edges__(line_graph_colors, self.G)

            num_classes = len(pe)
            if num_classes < ncolors:
                self.parallel_edges = pe
                ncolors = num_classes
