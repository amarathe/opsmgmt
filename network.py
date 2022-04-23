import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from attrs import define


@define
class NodeTimings:
    es: int = 0
    ef: int = 0
    ls: int = 0
    lf: int = 0
    val: int = 0


class GraphVisualization:
    """ Create a graph of dependencies and calculate early start & finish, late start & finish
        and slack. Graph must be acyclic, directed, unweighted with > 1 nodes
    """

    def __init__(self):
        self.edges = []
        self.node2timings: dict[str, NodeTimings] = {}

    def populate_edges(self, edges: list[(str, str)]):
        """Populate graph edges
            Args: edge: list of directed edges [(pre:post)]
        """
        self.edges = edges

    def populate_weights(self, nodes: dict[str, int]):
        for node in nodes:
            self.node2timings[node] = NodeTimings(val=nodes[node])

    def populate_timings(self):
        """ Calculates timings of each node
            Algorithm: BFS
        """
        pre2post = defaultdict(list)
        post2pre = defaultdict(list)
        nodeorder = defaultdict(lambda: 0)
        for pre, post in self.edges:
            pre2post[pre].append(post)
            post2pre[post].append(pre)
            nodeorder[post] += 1
        bfs = list(set(x[0] for x in self.edges) - set(x[1]
                   for x in self.edges))
        while bfs:
            newbfs = []
            for node in bfs:
                # All dependencies satisfied, so safe to calculate minimum
                timings = self.node2timings[node]
                timings.es = max(
                    (self.node2timings[prev].ef for prev in post2pre[node]), default=0)
                timings.ef = timings.es + timings.val
                for post in pre2post[node]:
                    nodeorder[post] -= 1
                    if nodeorder[post] == 0:
                        newbfs.append(post)
                    elif nodeorder[post] < 0:
                        raise Exception("ERROR! Detected cycle on node: " + node)
            #Already tracking duplicates by keeping track of nodeorder
            #So Just remove duplicates from bfs
            bfs = newbfs

        # Do another BFS from last node to first to calculate ls, lf
        nodeorder = defaultdict(lambda: 0)
        for pre, _ in self.edges:
            nodeorder[pre] += 1
        bfs = list(set(x[1] for x in self.edges) - set(x[0]
                   for x in self.edges))
        while bfs:
            newbfs = []
            for node in bfs:
                timings = self.node2timings[node]
                timings.lf = min(
                    (self.node2timings[post].ls for post in pre2post[node]),
                    default=timings.ef)
                timings.ls = timings.lf - timings.val
                for pre in post2pre[node]:
                    nodeorder[pre] -= 1
                    if nodeorder[pre] == 0:
                        # All dependencies satisfied, so safe to calculate minimum
                        newbfs.append(pre)
                    elif nodeorder[pre] < 0:
                        raise Exception(
                            "ERROR! Detected cycle on node during reverse: " + pre)
            bfs = list(set(newbfs))

    def calc_pos(self) -> dict[str, list[int]]:
        """Calculates positions of graph nodes.
            Algorithm: Use BFS based on predecessors to arrange nodes from left to right
             Secondary sorting (Stack bottom to top) based on alphabetical order for prettyness
           Returns: Dictionary of {nodename : [x,y]}
        """
        pos = {}
        pre2post = defaultdict(list)
        nodeorder = defaultdict(lambda: 0)
        for pre, post in self.edges:
            pre2post[pre].append(post)
            nodeorder[post] += 1
        bfs = list(set(x[0] for x in self.edges) - set(x[1]
                   for x in self.edges))
        # Positions to plot.
        x, y = 0, 0
        while bfs:
            newbfs = []
            # Sort for prettyness (bottom to top is alphabetical)
            for node in sorted(bfs):
            # All dependencies satisfied, so draw the node
                pos[node] = [x, y]
                y += 1
                for post in pre2post[node]:
                    nodeorder[post] -= 1
                    if nodeorder[post] == 0:
                        newbfs.append(post)
                    elif nodeorder[post] < 0:
                        raise Exception("ERROR! Detected cycle on node: " + node)
            bfs = list(set(newbfs))
            x, y = x + 1, 0
        return pos

    def labelstartfinish(self, pos):
        """ Label early start/finish and late start/finish under node
        """
        for node, (x, y) in pos.items():
            timings = self.node2timings.get(node, None)
            #If no timings given, don't calculate/display anything
            if timings is None: return
            plt.text(
                x - 0.1,
                y - 0.25,
                s=timings.es,
                bbox=dict(
                    facecolor='red',
                    alpha=0.5),
                horizontalalignment='center')
            plt.text(
                x + 0.1,
                y - 0.25,
                s=timings.ef,
                bbox=dict(
                    facecolor='red',
                    alpha=0.5),
                horizontalalignment='center')
            plt.text(
                x - 0.1,
                y - 0.5,
                s=timings.ls,
                bbox=dict(
                    facecolor='red',
                    alpha=0.5),
                horizontalalignment='center')
            plt.text(
                x + 0.1,
                y - 0.5,
                s=timings.lf,
                bbox=dict(
                    facecolor='red',
                    alpha=0.5),
                horizontalalignment='center')
            plt.text(
                x,
                y - 0.75,
                s="Slack: " + str(timings.ls - timings.es),
                bbox=dict(
                    facecolor='orange',
                    alpha=0.5),
                horizontalalignment='left')

    def visualize(self):

        G3 = nx.Graph()
        G3.add_edges_from(self.edges)

        pos = self.calc_pos()

        nx.draw(
            G3,
            pos,
            node_size=500,
            alpha=0.9,
            arrowstyle='-|>',
            arrows=True,
            labels={node: node for node in G3.nodes()}
        )
        self.labelstartfinish(pos)
        plt.show()
