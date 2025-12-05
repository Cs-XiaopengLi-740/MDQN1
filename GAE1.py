import igraph as ig
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def influential_node_detection(adj_matrix):
    """
    算法1：最具影响力节点识别算法

    Args:
        adj_matrix (np.ndarray): 网络的邻接矩阵 (n x n)。

    Returns:
        int: 网络中最具影响力节点的ID。
    """
    n = adj_matrix.shape[0]
    row_sums = adj_matrix.sum(axis=1, keepdims=True)
    # 处理可能出现的除以零的情况
    P = np.where(row_sums > 0, adj_matrix / row_sums, np.zeros_like(adj_matrix))
    W = np.ones((1, n))
    m = 100  # 设置一个较大的m值来近似无穷大

    P_m = np.linalg.matrix_power(P, m)
    W_star = np.dot(W, P_m)
    influential_node_id = np.argmax(W_star)
    return influential_node_id

def adjacency_matrix_reconstruction(adj_matrix):
    """
    算法2：邻接矩阵重建算法

    Args:
        adj_matrix (np.ndarray): 网络的邻接矩阵 (n x n)。

    Returns:
        np.ndarray: 重建后的邻接矩阵 (n x n)。
    """
    n = adj_matrix.shape[0]
    leader_node = influential_node_detection(adj_matrix)
    reconstructed_matrix = np.zeros((n, n), dtype=int)
    ordered_nodes = [leader_node]
    remaining_nodes = list(range(n))
    remaining_nodes.remove(leader_node)

    while remaining_nodes:
        current_node = ordered_nodes[-1]
        min_distance = float('inf')
        closest_neighbor = -1

        for neighbor in remaining_nodes:
            distance = euclidean(adj_matrix[current_node], adj_matrix[neighbor])
            if distance < min_distance:
                min_distance = distance
                closest_neighbor = neighbor

        ordered_nodes.append(closest_neighbor)
        remaining_nodes.remove(closest_neighbor)

    # 构建重建后的邻接矩阵
    for i in range(n):
        for j in range(n):
            if i != j:
                u, v = ordered_nodes[i], ordered_nodes[j]
                reconstructed_matrix[i, j] = adj_matrix[u, v]
    return reconstructed_matrix

# 改进的GAE模型实现，加入图卷积层
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class ImprovedGAE(nn.Module):
    def __init__(self, nfeat, nhid, nlatent, dropout=0.5):
        super(ImprovedGAE, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nlatent)
        self.dropout = dropout
        self.relu = nn.ReLU()

    def encode(self, x, adj):
        h = self.relu(self.gc1(x, adj))
        h = nn.Dropout(self.dropout)(h)
        z = self.gc2(h, adj)
        return z

    def decode(self, z):
        # 内积解码器
        adj_rec = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_rec

    def forward(self, x, adj):
        z = self.encode(x, adj)
        adj_rec = self.decode(z)
        return adj_rec, z

def preprocess_adj(adj):
    """对称归一化邻接矩阵"""
    adj = adj + np.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def community_detection_gae(reconstructed_matrix, n_clusters, q=5, 
                            hidden_dim=64, z_dim=32, epochs=200, lr=0.001, 
                            dropout=0.3, early_stopping=10):
    """
    算法3：基于GAE的社区检测算法 (增强版)

    Args:
        reconstructed_matrix (np.ndarray): 重建后的邻接矩阵 (n x n)。
        n_clusters (int): 期望的社区数量 (k)。
        q (int): 每条记录的大小。
        hidden_dim (int): GAE隐藏层维度。
        z_dim (int): GAE编码后的特征维度。
        epochs (int): 训练轮数。
        lr (float): 学习率。
        dropout (float): Dropout率。
        early_stopping (int): 早停耐心值。

    Returns:
        np.ndarray: 节点的社区分配标签。
    """
    n = reconstructed_matrix.shape[0]
    
    # 预处理邻接矩阵
    adj_norm = preprocess_adj(reconstructed_matrix)
    
    # 转换为PyTorch张量
    adj = torch.FloatTensor(adj_norm)
    features = torch.FloatTensor(np.eye(n))  # 使用单位矩阵作为初始特征
    
    # 初始化GAE模型
    model = ImprovedGAE(nfeat=n, nhid=hidden_dim, nlatent=z_dim, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    # 训练GAE
    best_loss = float('inf')
    patience = 0
    z_best = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        adj_rec, z = model(features, adj)
        loss = criterion(adj_rec, adj)
        loss.backward()
        optimizer.step()
        
        # 早停机制
        if loss.item() < best_loss:
            best_loss = loss.item()
            z_best = z.detach().numpy()
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    # 使用KMeans进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=20)
    cluster_labels = kmeans.fit_predict(z_best)
    
    return cluster_labels

def visualize_communities(g, labels, title="Community Detection Results"):
    """可视化社区检测结果"""
    layout = g.layout_fruchterman_reingold()
    visual_style = {
        "vertex_size": 20,
        "vertex_color": labels,
        "vertex_label": g.vs["id"] if "id" in g.vs.attributes() else None,
        "layout": layout,
        "bbox": (800, 600),
        "margin": 50,
        "main": title
    }
    return ig.plot(g, **visual_style)

if __name__ == '__main__':
    # 加载数据集
    try:
        g = ig.Graph.Read_GML("datasets/football.gml")
    except FileNotFoundError:
        print("数据集文件未找到，请确保路径正确。")
        exit()
        
    adj_matrix = np.array(g.get_adjacency().data)
    true_partition = [7, 0, 2, 3, 7, 3, 2, 8, 8, 7, 3, 10, 6, 2, 6, 2, 7, 9, 6, 1, 9, 8, 8, 7, 10, 0, 6, 9, 11, 1, 1, 6, 2, 0, 6, 1, 5, 0, 6, 2, 3, 7, 5, 6, 4, 0, 11, 2, 4, 11, 10, 8, 3, 11, 6, 1, 9, 4, 11, 10, 2, 6, 9, 10, 2, 9, 4, 11, 8, 10, 9, 6, 3, 11, 3, 4, 9, 8, 8, 1, 5, 3, 5, 11, 3, 6, 4, 9, 11, 0, 5, 4, 4, 7, 1, 9, 9, 10, 3, 6, 2, 1, 3, 0, 7, 0, 2, 3, 8, 0, 4, 8, 4, 9, 11]
    n_nodes = adj_matrix.shape[0]
    n_communities_true = len(set(true_partition))

    # 设置参数
    q_value = 5  # 记录大小
    hidden_dim = 64  # GAE隐藏层维度
    z_dim = 32  # GAE编码后的特征维度
    epochs = 200  # 训练轮数
    learning_rate = 0.001  # 学习率
    dropout = 0.3  # Dropout率
    early_stopping = 15  # 早停耐心值

    # 创建保存结果的目录
    if not os.path.exists("results"):
        os.makedirs("results")

    # 步骤1：最具影响力节点识别
    leader_node = influential_node_detection(adj_matrix)
    print(f"最具影响力节点的ID: {leader_node}")

    # 步骤2：邻接矩阵重建
    reconstructed_adj = adjacency_matrix_reconstruction(adj_matrix)
    print("重建后的邻接矩阵 (部分):\n", reconstructed_adj[:5, :5])

    # 保存重建的邻接矩阵
    np.save("results/reconstructed_adj.npy", reconstructed_adj)

    # 步骤3：基于GAE的社区检测
    try:
        print("开始社区检测...")
        predicted_labels = community_detection_gae(
            reconstructed_adj, 
            n_communities_true, 
            q=q_value,
            hidden_dim=hidden_dim,
            z_dim=z_dim,
            epochs=epochs,
            lr=learning_rate,
            dropout=dropout,
            early_stopping=early_stopping
        )
        print("预测的社区标签:\n", predicted_labels)

        # 评估结果
        nmi = normalized_mutual_info_score(true_partition, predicted_labels)
        ari = adjusted_rand_score(true_partition, predicted_labels)
        print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
        print(f"Adjusted Rand Index (ARI): {ari:.4f}")

        # 计算模块度
        partition = ig.VertexClustering(g, membership=predicted_labels)
        modularity = g.modularity(partition)
        print(f"Modularity: {modularity:.4f}")

        # 可视化结果
        plot = visualize_communities(g, predicted_labels, 
                                    f"Community Detection (NMI: {nmi:.2f}, Modularity: {modularity:.2f})")
        plot.save("results/community_detection.png")
        
        # 保存结果
        with open("results/metrics.txt", "w") as f:
            f.write(f"NMI: {nmi:.4f}\n")
            f.write(f"ARI: {ari:.4f}\n")
            f.write(f"Modularity: {modularity:.4f}\n")
            f.write(f"Predicted labels: {list(predicted_labels)}\n")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()    
