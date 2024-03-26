import random
import copy
import sys
import tkinter
import threading
from functools import reduce

# 蚁群算法参数
(ALPHA, BETA, RHO, Q) = (1.0, 2.0, 0.5, 100.0)
# 城市数，蚁群
(city_num, ant_num) = (14, 50)
# 城市坐标
distance_x = [78, 278, 600, 700, 330, 550, 230, 380, 450, 720, 150, 330, 532, 700]
distance_y = [170, 100, 78, 151, 200, 220, 280, 300, 305, 300, 500, 550, 525, 500]

# 城市距离和信息素
distance_graph = [[0, 72, 98, 133, 54, 105, 55, 83, 79, 140, 26, 50, 121, 115],
                  [72, 0, 56, 97, 18, 69, 52, 74, 66, 111, 98, 90, 108, 102],
                  [98, 56, 0, 123, 44, 95, 78, 100, 92, 137, 124, 116, 134, 128],
                  [133, 97, 123, 0, 79, 28, 113, 84, 55, 70, 108, 84, 97, 91],
                  [54, 18, 44, 79, 0, 51, 34, 56, 48, 93, 80, 72, 90, 84],
                  [105, 69, 95, 28, 51, 0, 85, 56, 27, 42, 80, 56, 69, 63],
                  [55, 52, 78, 113, 34, 85, 0, 36, 65, 126, 62, 38, 107, 101],
                  [83, 74, 100, 84, 56, 56, 36, 0, 29, 90, 57, 33, 71, 65],
                  [79, 66, 92, 55, 48, 27, 65, 29, 0, 61, 53, 29, 42, 36],
                  [140, 111, 137, 70, 93, 42, 126, 90, 61, 0, 114, 90, 72, 25],
                  [26, 98, 124, 108, 80, 80, 62, 57, 53, 114, 0, 24, 95, 89],
                  [50, 90, 116, 84, 72, 56, 38, 33, 29, 90, 24, 0, 71, 65],
                  [121, 108, 134, 97, 90, 69, 107, 71, 42, 72, 95, 71, 0, 47],
                  [115, 102, 128, 91, 84, 63, 101, 65, 36, 25, 89, 65, 47, 0]]
pheromone_graph = [[1.0 for col in range(city_num)] for raw in range(city_num)]
x = []
y = []


# 蚁群算法
class Ant(object):
    # 初始化
    def __init__(self, ID):
        self.ID = ID  # ID
        self.__clean_data()

    # 初始数据
    def __clean_data(self):
        self.path = []
        self.total_distance = 0.0  # 当前路径的总距离
        self.move_count = 0  # 移动次数
        self.current_city = -1  # 当前停留的城市
        self.open_table_city = [True for i in range(city_num)]  # 探索城市的状态
        city_index = random.randint(0, city_num - 1)  # 随机初始出生点
        self.current_city = city_index
        self.path.append(city_index)
        self.open_table_city[city_index] = False
        self.move_count = 1

    # 选择下一个城市
    def __choice_next_city(self):
        next_city = -1
        select_citys_prob = [0.0 for i in range(city_num)]  # 存储去下个城市的概率
        total_prob = 0.0
        # 获取去下一个城市的概率
        for i in range(city_num):
            if self.open_table_city[i]:
                try:
                    # 计算概率：与信息素浓度成正比，与距离成反比
                    select_citys_prob[i] = pow(pheromone_graph[self.current_city][i], ALPHA) * pow(
                        (1.0 / distance_graph[self.current_city][i]), BETA)
                    total_prob += select_citys_prob[i]
                except ZeroDivisionError as e:
                    print('Ant ID: {ID}, current city: {current}, target city: {target}'.format(ID=self.ID,
                                                                                                current=self.current_city,
                                                                                                target=i))
                    sys.exit(1)
        # 轮盘选择城市
        if total_prob > 0.0:
            # 产生一个随机概率,0.0-total_prob
            temp_prob = random.uniform(0.0, total_prob)
            for i in range(city_num):
                if self.open_table_city[i]:  # 轮次相减
                    temp_prob -= select_citys_prob[i]
                    if temp_prob < 0.0:
                        next_city = i
                        break
        if next_city == -1:
            next_city = random.randint(0, city_num - 1)
            while not (self.open_table_city[next_city]):  # self.open_table_city[next_city]==False,说明已经遍历过了
                next_city = random.randint(0, city_num - 1)
        # 返回下一个城市序号
        return next_city

    # 计算路径总距离
    def __cal_total_distance(self):
        temp_distance = 0.0
        for i in range(1, city_num):
            start, end = self.path[i], self.path[i - 1]
            temp_distance += distance_graph[start][end]
        # 回路
        end = self.path[0]
        temp_distance += distance_graph[start][end]
        self.total_distance = temp_distance

    # 移动操作
    def __move(self, next_city):
        self.path.append(next_city)
        self.open_table_city[next_city] = False
        self.total_distance += distance_graph[self.current_city][next_city]
        self.current_city = next_city
        self.move_count += 1

    # 搜索路径
    def search_path(self):
        # 初始化数据
        self.__clean_data()
        # 搜素路径，遍历完所有城市为止
        while self.move_count < city_num:
            # 移动到下一个城市
            next_city = self.__choice_next_city()
            self.__move(next_city)
        # 计算路径总长度
        self.__cal_total_distance()


# TSP问题
class TSP(object):
    def __init__(self, root, width=900, height=700, n=city_num):
        # 创建画布
        self.root = root
        self.width = width
        self.height = height
        # 城市数目初始化为city_num
        self.n = n
        self.canvas = tkinter.Canvas(
            root,
            width=self.width,
            height=self.height,
            bg="#EBEBEB",  # 背景颜色
            xscrollincrement=1,
            yscrollincrement=1
        )
        self.canvas.pack(expand=tkinter.YES, fill=tkinter.BOTH)
        self.title("a:初始化 b:开始搜索 c:停止搜索 d:退出程序")
        self.__r = 5
        self.__lock = threading.RLock()  # 线程锁
        self.__bindEvents()
        self.new()
        # 计算城市之间的距离

    # 按键响应程序
    def __bindEvents(self):
        self.root.bind("a", self.new)  # 初始化
        self.root.bind("b", self.search_path)  # 开始搜索
        self.root.bind("c", self.stop)  # 停止搜索
        self.root.bind("d", self.quite)  # 退出程序

    # 更改标题
    def title(self, s):
        self.root.title(s)

    # 初始化
    def new(self, evt=None):
        self.__lock.acquire()  # 停止线程
        self.__running = False
        self.__lock.release()
        self.clear()  # 清除信息
        self.nodes = []  # 节点坐标
        self.nodes2 = []  # 节点对象
        # 初始化城市节点
        for i in range(city_num):
            # 在画布上随机初始坐标
            x = distance_x[i]
            y = distance_y[i]
            self.nodes.append((x, y))
            # 生成节点椭圆，半径为self.__r
            node = self.canvas.create_oval(x - 15, y - 15, x + 15, y + 15,
                                           fill="#7FFF00",  # 填充绿色
                                           outline="#000000",  # 轮廓黑色
                                           tags="node",
                                           )
            self.nodes2.append(node)
            # 显示坐标
            self.canvas.create_text(x, y,  # 使用create_text方法在处绘制文字
                                    text=i + 1,  # 所绘制文字的内容
                                    fill='black'  # 所绘制文字的颜色为黑色
                                    )
        # 初始城市之间的距离和信息素
        for i in range(city_num):
            for j in range(city_num):
                pheromone_graph[i][j] = 1.0
        self.ants = [Ant(ID) for ID in range(ant_num)]  # 初始蚁群
        self.best_ant = Ant(-1)  # 初始最优解
        self.best_ant.total_distance = 1 << 31  # 初始最大距离
        self.iter = 1  # 初始化迭代次数

    # 将节点按order顺序连线
    def line(self, order):
        # 删除原线
        self.canvas.delete("line")

        def line2(i1, i2):
            p1, p2 = self.nodes[i1], self.nodes[i2]
            self.canvas.create_line(p1, p2, fill="#000000", tags="line")
            return i2

        # order[-1]为初始值
        reduce(line2, order, order[-1])

    # 清除画布
    def clear(self):
        for item in self.canvas.find_all():
            self.canvas.delete(item)

    # 退出程序
    def quite(self, evt):
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()
        self.root.destroy()
        print(u"\n程序已退出...")
        sys.exit()

    # 停止搜索
    def stop(self, evt):
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()

    # 开始搜索
    def search_path(self, evt=None):
        # 开启线程
        self.__lock.acquire()
        self.__running = True
        self.__lock.release()
        while self.__running:
            # 遍历每一只蚂蚁
            for ant in self.ants:
                # 搜索一条路径
                ant.search_path()
                # 与当前最优蚂蚁比较
                if ant.total_distance < self.best_ant.total_distance:
                    # 更新最优解
                    self.best_ant = copy.deepcopy(ant)
            # 更新信息素
            self.__update_pheromone_gragh()
            print(u"迭代次数：", self.iter, u"最佳路径总距离：",
                  int(self.best_ant.total_distance))
            time = self.best_ant.total_distance / 50
            print("一次整体配送所花时间为：{:.4f}".format(time))
            x.append(self.iter)
            y.append(time)
            # 连线
            self.line(self.best_ant.path)
            # 设置标题
            self.title("a:随机初始 b:开始搜索 c:停止搜索 d:退出程序 迭代次数: %d" % self.iter)
            # 更新画布
            self.canvas.update()
            self.iter += 1

    # 更新信息素
    def __update_pheromone_gragh(self):
        # 获取每只蚂蚁在其路径上留下的信息素
        temp_pheromone = [[0.0 for col in range(city_num)] for raw in range(city_num)]
        for ant in self.ants:
            for i in range(1, city_num):
                start, end = ant.path[i - 1], ant.path[i]
                # 在路径上的每两个相邻城市间留下信息素，与路径总距离反比
                temp_pheromone[start][end] += Q / ant.total_distance
                temp_pheromone[end][start] = temp_pheromone[start][end]
        # 更新所有城市之间的信息素，旧信息素衰减加上新迭代信息素
        for i in range(city_num):
            for j in range(city_num):
                pheromone_graph[i][j] = pheromone_graph[i][j] * RHO + temp_pheromone[i][j]

    # 主循环
    def mainloop(self):
        self.root.mainloop()


# 程序的入口处
if __name__ == '__main__':
    TSP(tkinter.Tk()).mainloop()
