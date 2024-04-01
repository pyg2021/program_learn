import numpy as np

def positional_encoding(max_position, embedding_dim):
    """
    正弦余弦编码（Sine-Cosine Encoding）的实现函数
    
    参数：
    max_position: int，最大位置
    embedding_dim: int，编码向量的维度
    
    返回：
    position_encoding: numpy array，位置编码矩阵，shape为(max_position, embedding_dim)
    """
    position_encoding = np.zeros((max_position, embedding_dim))
    for pos in range(max_position):
        for i in range(embedding_dim):
            if i % 2 == 0:
                position_encoding[pos, i] = np.sin(pos / (10000 ** (2 * i / embedding_dim)))
            else:
                position_encoding[pos, i] = np.cos(pos / (10000 ** (2 * i / embedding_dim)))
    return position_encoding

def embedding_lookup(vocabulary_size, embedding_dim):
    """
    嵌入层的实现函数，用于将单词映射为固定维度的向量表示
    
    参数：
    vocabulary_size: int，词汇表的大小
    embedding_dim: int，嵌入向量的维度
    
    返回：
    embedding_matrix: numpy array，嵌入矩阵，shape为(vocabulary_size, embedding_dim)
    """
    embedding_matrix = np.random.rand(vocabulary_size, embedding_dim)
    return embedding_matrix

# 示例：将位置编码和嵌入向量结合起来
max_position = 10  # 假设文本序列的最大长度为10
vocabulary_size = 1000  # 假设词汇表的大小为1000
embedding_dim = 50  # 设定嵌入向量的维度为50

# 生成位置编码矩阵和嵌入矩阵
pos_encoding = positional_encoding(max_position, embedding_dim)
embedding_matrix = embedding_lookup(vocabulary_size, embedding_dim)

# 将位置编码和嵌入向量相加
combined_embedding = pos_encoding + embedding_matrix

# 输出前10个位置的编码向量和对应的嵌入向量
for pos in range(max_position):
    print("Position", pos, "的位置编码:", pos_encoding[pos])
    print("Position", pos, "的嵌入向量:", embedding_matrix[pos])
    print("Position", pos, "的结合向量:", combined_embedding[pos])
