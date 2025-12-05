import networkx as nx
import community.community_louvain as community_louvain
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix

def read_data(edge_file, label_file):
    """读取边数据和标签数据"""
    G = nx.Graph()
    with open(edge_file, 'r') as f:
        for line in f:
            u, v, *rest = line.strip().split()
            G.add_edge(u, v)
    
    true_labels = {}
    with open(label_file, 'r') as f:
        for line in f:
            node, label, *rest = line.strip().split()
            true_labels[node] = label
    
    return G, true_labels

def compute_f1_score(true_labels, community_labels):
    """计算F1分数"""
    all_nodes = set(true_labels) | set(community_labels)
    true_labels_full = {n: true_labels.get(n, -1) for n in all_nodes}
    comm_labels_full = {n: community_labels.get(n, -1) for n in all_nodes}
    
    nodes = sorted(all_nodes)
    true_list = [true_labels_full[n] for n in nodes]
    comm_list = [comm_labels_full[n] for n in nodes]
    
    cm = contingency_matrix(true_list, comm_list, sparse=True)
    f1_scores = []
    for i in range(cm.shape[0]):
        dist = cm[i].toarray()[0]
        if dist.sum() == 0:
            continue
        best_j = np.argmax(dist)
        precision = dist[best_j] / cm[:, best_j].toarray().sum()
        recall = dist[best_j] / dist.sum()
        if precision + recall > 0:
            f1_scores.append(2 * precision * recall / (precision + recall))
    return np.mean(f1_scores) if f1_scores else 0

def compute_con(G, partition):
    """计算 CON 值"""
    # 先反转 partition：{community_id: [node, ...], ...}
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)
    
    con_vals = []
    all_nodes = set(G.nodes())
    for nodes in communities.values():
        nodes_set = set(nodes)
        # 社区 C 的体积 = C 内所有节点的度之和
        vol_c = sum(G.degree(n) for n in nodes_set)
        # 补集体积
        vol_comp = sum(G.degree(n) for n in all_nodes - nodes_set)
        # 社区与补集之间的边数
        inter = sum(
            1
            for u in nodes_set
            for v in G.neighbors(u)
            if v not in nodes_set
        )
        min_vol = min(vol_c, vol_comp)
        if min_vol > 0:
            con_vals.append(inter / min_vol)
    
    return np.mean(con_vals) if con_vals else 0

def main():
    edge_file = "datasets/orkutedge.txt"
    label_file = "datasets/orkutlabel.txt"
    
    G, true_labels = read_data(edge_file, label_file)
    print(f"图节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
    
    partition = community_louvain.best_partition(G)
    print(f"发现社区数量: {len(set(partition.values()))}")
    
    # NMI
    true_num = {n: int(l) for n, l in true_labels.items()}
    comm_num = {n: c for n, c in partition.items()}
    all_nodes = sorted(set(true_num) | set(comm_num))
    nmi = normalized_mutual_info_score(
        [true_num.get(n, -1) for n in all_nodes],
        [comm_num.get(n, -1) for n in all_nodes]
    )
    print(f"NMI值: {nmi:.4f}")
    
    # F1
    f1 = compute_f1_score(true_labels, partition)
    print(f"F1值: {f1:.4f}")
    
    # CON
    con = compute_con(G, partition)
    print(f"CON值: {con:.4f}")
    
    # 输出结果
    with open("community_assignments.txt", "w") as f:
        for node, comm in partition.items():
            f.write(f"{node} {comm}\n")

if __name__ == "__main__":
    main()
