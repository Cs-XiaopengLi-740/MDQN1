import networkx as nx
import leidenalg as la
import igraph as ig
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
import itertools

def read_data(edge_file, label_file):
    """读取边数据和标签数据"""
    # 读取边数据并构建图
    G = nx.Graph()
    with open(edge_file, 'r') as f:
        for line in f:
            nodes = line.strip().split()
            if len(nodes) >= 2:
                u, v = nodes[0], nodes[1]
                G.add_edge(u, v)
    
    # 读取节点标签
    true_labels = {}
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                node, label = parts[0], parts[1]
                true_labels[node] = label
    
    return G, true_labels

def compute_f1_score(true_labels, community_labels):
    """计算F1分数"""
    # 确保所有节点都有标签
    all_nodes = set(true_labels.keys()).union(set(community_labels.keys()))
    true_labels = {node: true_labels.get(node, -1) for node in all_nodes}
    community_labels = {node: community_labels.get(node, -1) for node in all_nodes}
    
    # 转换为列表
    true_list = [true_labels[node] for node in all_nodes]
    comm_list = [community_labels[node] for node in all_nodes]
    
    # 构建列联表
    cm = contingency_matrix(true_list, comm_list, sparse=True)
    
    f1_scores = []
    # 对每个真实社区计算F1
    for i in range(cm.shape[0]):
        # 找到该真实社区在预测社区中的分布
        dist = cm[i].toarray()[0]
        if np.sum(dist) == 0:
            continue
        
        # 找到最佳匹配的预测社区
        best_j = np.argmax(dist)
        precision = dist[best_j] / np.sum(cm[:, best_j].toarray())
        recall = dist[best_j] / np.sum(dist)
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            f1_scores.append(f1)
    
    # 返回平均F1
    return np.mean(f1_scores) if f1_scores else 0

def compute_con(G, partition):
    """计算CON值"""
    con_values = []
    total_nodes = set(G.nodes())
    
    # 转换partition格式为社区ID到节点集合的映射
    community_to_nodes = {}
    for node, comm_id in partition.items():
        if comm_id not in community_to_nodes:
            community_to_nodes[comm_id] = set()
        community_to_nodes[comm_id].add(node)
    
    for community_id, nodes in community_to_nodes.items():
        # 计算社区C的体积
        vol_c = sum(G.degree(node) for node in nodes)
        # 计算补集的体积
        vol_complement = sum(G.degree(node) for node in total_nodes - nodes)
        
        # 计算社区C与补集之间的边数
        inter_edges = 0
        for u in nodes:
            for v in G.neighbors(u):
                if v not in nodes:
                    inter_edges += 1
        
        # 计算CON值
        min_vol = min(vol_c, vol_complement)
        if min_vol > 0:
            con = inter_edges / min_vol
            con_values.append(con)
    
    # 返回所有社区的平均CON值
    return np.mean(con_values) if con_values else 0

def main():
    # 数据集路径
    edge_file = "datasets/michedge.txt"
    label_file = "datasets/michlabel.txt"
    
    # 读取数据
    G_nx, true_labels = read_data(edge_file, label_file)
    print(f"图节点数: {G_nx.number_of_nodes()}, 边数: {G_nx.number_of_edges()}")
    
    # 转换NetworkX图为igraph图
    G_ig = ig.Graph.from_networkx(G_nx)
    
    # 使用Leiden算法进行社区发现
    partition = la.find_partition(G_ig, la.ModularityVertexPartition)
    
    # 转换分区结果为节点到社区ID的映射
    node_to_community = {}
    for comm_id, nodes in enumerate(partition):
        for node_idx in nodes:
            node_name = G_ig.vs[node_idx]['_nx_name']
            node_to_community[node_name] = comm_id
    
    print(f"发现社区数量: {len(partition)}")
    
    # 计算NMI
    # 确保标签是数值类型
    true_numeric = {node: int(label) for node, label in true_labels.items()}
    comm_numeric = {node: int(comm) for node, comm in node_to_community.items()}
    
    # 确保所有节点都包含在内
    all_nodes = set(true_numeric.keys()).union(set(comm_numeric.keys()))
    true_list = [true_numeric.get(node, -1) for node in all_nodes]
    comm_list = [comm_numeric.get(node, -1) for node in all_nodes]
    
    nmi = normalized_mutual_info_score(true_list, comm_list)
    print(f"NMI值: {nmi:.4f}")
    
    # 计算F1分数
    f1 = compute_f1_score(true_labels, node_to_community)
    print(f"F1值: {f1:.4f}")
    
    # 计算CON值
    con = compute_con(G_nx, node_to_community)
    print(f"CON值: {con:.4f}")
    
    # 输出社区分配结果
    with open("community_assignments.txt", "w") as f:
        for node, community in node_to_community.items():
            f.write(f"{node} {community}\n")

if __name__ == "__main__":
    main()
