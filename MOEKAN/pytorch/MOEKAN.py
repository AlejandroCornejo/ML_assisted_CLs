

MOEKAN = [3, 2, 1, 1]

edges = []

for layer in range(len(MOEKAN) - 1):
    n_in = MOEKAN[layer]
    n_out = MOEKAN[layer + 1]

    layer_edges = []
    for i in range(n_in):
        row = []
        for j in range(n_out):
            row.append([i, j])  # or Edge(i,j)
        layer_edges.append(row)
    edges.append(layer_edges)


print(edges[0][0][0])
# print(edges[2])