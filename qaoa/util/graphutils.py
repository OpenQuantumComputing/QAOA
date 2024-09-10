import networkx as nx


def map_colors_to_edges(line_graph_colors, original_graph):
    edge_colors = {}
    for edge in original_graph.edges():
        # Line graph uses sorted tuples for edges
        edge_in_line_graph = tuple(sorted(edge))
        edge_colors[edge] = line_graph_colors[edge_in_line_graph]
    return edge_colors


def group_edges_by_color(edge_colors):
    color_groups = {}
    for edge, color in edge_colors.items():
        if color not in color_groups:
            color_groups[color] = []
        color_groups[color].append(edge)
    return color_groups


def minimum_edge_coloring(G, repetitions=100):
    #
    # a graph G
    # returns minimum edge coloring, i.e., a dict containting the edges for each color
    # this can be used to minimize the depth needed to implement diagonal cost Hamiltonians
    # example output
    # { '3' : [('0', '1'), ('2', '7'), ('3', '9'), ('5', '6')]
    #   '1': [('0', '3'), ('1', '5'), ('7', '9')]
    #   '2': [('0', '9'), ('1', '6'), ('2', '5'), ('3', '8'), ('4', '7')]
    #   '0': [('1', '4'), ('2', '9'), ('3', '7'), ('5', '8')] }
    #

    ncolors = G.number_of_edges() + 1
    for _ in range(repetitions):
        # Convert the graph to its line graph
        line_G = nx.line_graph(G)

        # Apply greedy vertex coloring on the line graph
        line_graph_colors = nx.coloring.greedy_color(
            line_G, strategy="random_sequential"
        )

        # Map the result back to the original graph as edge colors
        es = map_colors_to_edges(line_graph_colors, G)

        # Group edges by their assigned color
        cs = group_edges_by_color(es)
        if len(cs) < ncolors:
            color_groups = cs
            edge_colors = es
            ncolors = len(cs)

    return color_groups, edge_colors
