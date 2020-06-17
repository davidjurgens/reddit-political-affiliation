import networkx as nx
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale

mpl.use('macosx')


def build_subreddit_commonality_graph(subreddits, subreddit_overlaps):
    nodes = [s.subreddit_name for s in subreddits]
    node_sizes = [float(s.comment_count) for s in subreddits]
    edge_widths = [float(o.user_overlap) for o in subreddit_overlaps]

    node_sizes = minmax_scale(node_sizes, (100, 2000))
    edge_widths = minmax_scale(edge_widths, (.1, 3))
    G = nx.Graph()
    G.add_nodes_from(nodes)

    for i, overlap in enumerate(subreddit_overlaps):
        if overlap.sub_one in nodes and overlap.sub_two in nodes:
            G.add_edge(overlap.sub_one, overlap.sub_two, weight=edge_widths[i])

    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes)
    nx.draw_networkx_edges(G, pos, width=edge_widths)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', font_color='w')

    plt.axis('off')
    plt.show()
