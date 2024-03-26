# 创建节点字典
set_nodes={"v1": 0, "v2": 1, "v3": 2, "v4": 3, "v5": 4, "v6": 5, "v7": 6, "v8": 7, "v9": 8,
           "v10": 9, "v11": 10, "v12": 11, "v13": 12, "v14": 13}
INF = 999
# 创建初始化距离矩阵
dis = ([[0, INF, INF, INF, 54, INF, 55, INF, INF, INF, 26, INF, INF, INF],
    [INF, 0, 56, INF, 18, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    [INF, 56, 0, INF, 44, INF, INF, INF, INF, INF, INF, INF, INF, INF],
    [INF, INF, INF, 0, INF, 28, INF, INF, INF, INF, INF, INF, INF, INF],
    [54, 18, 44, INF, 0, 51, 34, 56, 48, INF, INF, INF, INF, INF],
    [INF, INF, INF, 28, 51, 0, INF, INF, 27, 42, INF, INF, INF, INF],
    [55, INF, INF, INF, 34, INF, 0, 36, INF, INF, INF, 38, INF, INF],
    [INF, INF, INF, INF, 56, INF, 36, 0, 29, INF, INF, 33, INF, INF],
    [INF, INF, INF, INF, 48, 27, INF, 29, 0, 61, INF, 29, 42, 36],
    [INF, INF, INF, INF, INF, 42, INF, INF, 61, 0, INF, INF, INF, 25],
    [26, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0, 24, INF, INF],
    [INF, INF, INF, INF, INF, INF, 38, 33, 29, INF, 24, 0, INF, INF],
    [INF, INF, INF, INF, INF, INF, INF, INF, 42, INF, INF, INF, 0, 47],
    [INF, INF, INF, INF, INF, INF, INF, INF, 36, 25, INF, INF, 47, 0]])
num = 14
# 初始化一个矩阵记录父节点先令父节点为终点本身
parent=[[i for i in range(14)] for j in range(14)]
# 核心代码
# i为中间节点
for i in range(num):
    # j为起点
    for j in range(num):
        # k为终点
        for k in range(num):
            # 更新最短距离
            dis[j][k] = min(dis[j][k], dis[j][i]+dis[i][k])
            parent[j][k] = parent[j][i]
# 测试代码
if __name__ == '__main__':
    for i in range(num):
        # j为起点
        print("[", end='')
        for j in range(num):
            print( str(dis[i][j])+',', end='')
        print("],")
