import random


class SparseGrammar:
    def __init__(self, node_count, prefix, suffix):
        self._edges = [['' for _ in range(node_count)] for _ in range(node_count)]
        self._prefix = prefix
        self._suffix = suffix

    def add_edge(self, source_idx, dest_idx, char):
        self._edges[source_idx][dest_idx] = char

    def produce(self):
        yield self._prefix
        curr_idx = 0
        while True:
            edges = self._edges[curr_idx]
            destinations = [i for i in range(len(edges)) if edges[i] != '']
            if not any(destinations):
                yield self._suffix
                return

            dest_idx = destinations[random.randint(0, len(destinations) - 1)]
            yield edges[dest_idx]
            curr_idx = dest_idx

    def get_edge_labels(self):
        edge_labels = []
        for x in range(len(self._edges)):
            for y in range(len(self._edges)):
                if self._edges[x][y] != "" and self._edges[x][x] not in edge_labels:
                    edge_labels.append(self._edges[x][y])
        return edge_labels
