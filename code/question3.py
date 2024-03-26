import random
import copy
import sys
import tkinter
import threading
from functools import reduce

# 蚁群算法参数
(ALPHA, BETA, RHO, Q) = (1.0, 2.0, 0.5, 100.0)
# 城市数，蚁群
(city_num, ant_num) = (14, 100)
# 城市坐标
distance_x = [78, 278, 600, 700, 330, 550, 230, 380, 450, 720, 150, 330, 532, 700]
distance_y = [170, 100, 78, 151, 200, 200, 280, 300, 280, 300, 500, 550, 525, 500]
INF = 999999999
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
plant_path = [[0, 20, 76, 91, 38, 89, 52, 83, 79, 117, 26, 50, 84, 115],
              [20, 0, 56, 71, 18, 69, 52, 74, 66, 97, 46, 70, 99, 102],
              [76, 56, 0, 15, 44, 18, 78, 74, 45, 41, 98, 74, 87, 66],
              [91, 71, 15, 0, 59, 28, 93, 84, 55, 26, 108, 84, 97, 51],
              [38, 18, 44, 59, 0, 51, 34, 56, 48, 85, 60, 72, 81, 84],
              [89, 69, 18, 28, 51, 0, 85, 56, 27, 42, 80, 56, 69, 63],
              [52, 52, 78, 93, 34, 85, 0, 36, 65, 119, 26, 38, 61, 101],
              [83, 74, 74, 84, 56, 56, 36, 0, 29, 90, 57, 33, 25, 65],
              [79, 66, 45, 55, 48, 27, 65, 29, 0, 61, 53, 29, 42, 36],
              [117, 97, 41, 26, 85, 42, 119, 90, 61, 0, 114, 90, 72, 25],
              [26, 46, 98, 108, 60, 80, 26, 57, 53, 114, 0, 24, 58, 89],
              [50, 70, 74, 84, 72, 56, 38, 33, 29, 90, 24, 0, 34, 65],
              [84, 99, 87, 97, 81, 69, 61, 25, 42, 72, 58, 34, 0, 47],
              [115, 102, 66, 51, 84, 63, 101, 65, 36, 25, 89, 65, 47, 0]]
car_graph = [[0, INF, INF, INF, 54, INF, 55, INF, INF, INF, 26, INF, INF, INF],
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
             [INF, INF, INF, INF, INF, INF, INF, INF, 36, 25, INF, INF, 47, 0]]
plant_graph = [[0, 20, INF, INF, 54, INF, 55, INF, INF, INF, 26, INF, INF, INF],
               [20, 0, 56, INF, 18, INF, INF, INF, INF, INF, INF, INF, INF, INF],
               [INF, 56, 0, 15, 44, 18, INF, INF, INF, INF, INF, INF, INF, INF],
               [INF, INF, 15, 0, INF, 28, INF, INF, INF, 26, INF, INF, INF, INF],
               [54, 18, 44, INF, 0, 51, 34, 56, 48, INF, INF, INF, INF, INF],
               [INF, INF, 18, 28, 51, 0, INF, INF, 27, 42, INF, INF, INF, INF],
               [55, INF, INF, INF, 34, INF, 0, 36, INF, INF, 26, 38, INF, INF],
               [INF, INF, INF, INF, 56, INF, 36, 0, 29, INF, INF, 33, 25, INF],
               [INF, INF, INF, INF, 48, 27, INF, 29, 0, 61, INF, 29, 42, 36],
               [INF, INF, INF, 26, INF, 42, INF, INF, 61, 0, INF, INF, INF, 25],
               [26, INF, INF, INF, INF, INF, 26, INF, INF, INF, 0, 24, INF, INF],
               [INF, INF, INF, INF, INF, INF, 38, 33, 29, INF, 24, 0, 34, INF],
               [INF, INF, INF, INF, INF, INF, INF, 25, 42, INF, INF, 34, 0, 47],
               [INF, INF, INF, INF, INF, INF, INF, INF, 36, 25, INF, INF, 47, 0]]
pheromone_graph = [[1.0 for col in range(city_num)] for raw in range(city_num)]
plant_length = 0  # 无人机总路程
plant = [[0, 10], [3, 5], [13], [12], [11]]
car = [[1, 6], [2, 9], [9, 8], [8, 7], [7, 6]]
car_in = [True for i in range(city_num)]
# 已确定的飞行区
car_in[0] = False
car_in[10] = False
car_in[3] = False
car_in[2] = False
car_in[13] = False
car_in[12] = False
car_in[11] = False
xq1 = [12, 90, 24, 15, 70, 57, 198, 94, 30, 181, 36, 44, 42, 13]  # 需求
x1 = []
y1 = []


# ----------- 蚂蚁 -----------
class Ant(object):
    # 初始化
    def __init__(self, ID):
        # self.mg = None
        self.ID = ID  # ID
        self.__clean_data()  # 随机初始化出生点

    # 初始数据
    def __clean_data(self):
        self.path = []  # 当前蚂蚁的路径
        self.total_distance = 0.0  # 当前路径的总距离
        self.move_count = 0  # 移动次数
        self.current_city = -1  # 当前停留的城市
        self.open_table_city = [True for i in range(city_num)]  # 探索城市的状态
        city_index = 8  # 初始出生点
        self.current_city = city_index
        self.path.append(city_index)
        self.open_table_city[city_index] = False
        self.open_table_city[0] = False
        self.open_table_city[2] = False
        self.open_table_city[3] = False
        self.open_table_city[10] = False
        self.open_table_city[11] = False
        self.open_table_city[12] = False
        self.open_table_city[13] = False
        self.move_count = 6
        self.mg = 500

    # 选择下一个城市
    def __choice_next_city(self):
        next_city = -1
        select_citys_prob = [0.0 for i in range(city_num)]  # 存储去下个城市的概率
        total_prob = 0.0
        for i in range(city_num):
            if self.open_table_city[i]:
                try:
                    # 计算概率：与信息素浓度成正比，与距离成反比
                    select_citys_prob[i] = pow(pheromone_graph[self.current_city][i], ALPHA) * pow(
                        (1.0 / distance_graph[self.current_city][i]), BETA)
                    total_prob += select_citys_prob[i]
                except ZeroDivisionError as e:
                    print('Ant ID: {ID}, current city: {current}, target city: {target}'.
                          format(ID=self.ID, current=self.current_city, target=i))
                    sys.exit(1)
        # 轮盘选择城市
        if total_prob > 0.0:
            # 产生一个随机概率,0.0-total_prob
            temp_prob = random.uniform(0.0, total_prob)
            for i in range(city_num):
                if self.open_table_city[i] and (xq1[i] <= self.mg or i == 8):
                    # 轮次相减
                    temp_prob -= select_citys_prob[i]
                    if temp_prob < 0.0:
                        next_city = i
                        if i == 8:
                            self.mg = 500
                        else:
                            self.mg = self.mg - xq1[i]
                        break
        if (next_city == -1):
            # flag标志车的载重量是否用尽
            flag = 0
            for i in range(city_num):
                if xq1[i] <= self.mg and self.open_table_city[i] == True:
                    flag = 1
                    break
            # 载重用尽时返回物资点，重新满载物资
            if flag == 0:
                next_city = 8
                self.mg = 500
            else:
                next_city = random.randint(0, city_num - 1)
                while (self.open_table_city[next_city]) == False or xq1[next_city] > self.mg:
                    # 为False,说明已经遍历过了；物资需求超过剩余载重，说名此次运输应该结束
                    next_city = random.randint(0, city_num - 1)
                    # 返回下一个城市序号
                self.mg = self.mg - xq1[next_city]
        return next_city

    # 计算路径总距离
    def __cal_total_distance(self):
        temp_distance = 0.0
        for i in range(1, city_num - 6):
            start, end = self.path[i], self.path[i - 1]
            if car_in[start] == False or car_in[end] == False:
                temp_distance += plant_path[start][end]
            else:
                temp_distance += distance_graph[start][end]
        # 回路
        end = self.path[0]
        temp_distance += distance_graph[start][end]
        self.total_distance = temp_distance

    # 移动操作
    def __move(self, next_city):
        self.path.append(next_city)
        self.open_table_city[next_city] = False
        if car_in[self.current_city] == False or car_in[next_city] == False:
            self.total_distance += plant_path[self.current_city][next_city]
        else:
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


# ----------- TSP问题 ----------
class TSP(object):
    def __init__(self, root, width=900, height=700, n=city_num):
        # 创建画布
        self.root = root
        self.width = width
        self.height = height
        # 城市数目初始化为city_num
        self.n = n
        # tkinter.Canvas
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
        # 停止线程
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()
        self.clear()  # 清除信息
        self.nodes = []  # 节点坐标
        self.nodes2 = []  # 节点对象
        # 初始化城市节点
        for i in range(city_num):
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
            self.canvas.create_text(x, y,  # 使用create_text方法在坐标处绘制文字
                                    text=i + 1,  # 所绘制文字的内容
                                    font=5,
                                    fill='black'  # 所绘制文字的颜色为黑色
                                    )
        # 顺序连接城市
        # self.line(range(city_num))
        # 初始城市之间的距离和信息素
        for i in range(city_num):
            for j in range(city_num):
                pheromone_graph[i][j] = 1.0
        self.ants = [Ant(ID) for ID in range(ant_num)]  # 初始蚁群
        self.best_ant = Ant(-1)  # 初始最优解
        self.best_ant.total_distance = 1 << 31  # 初始最大距离
        self.iter = 1  # 初始化迭代次数

    # 将节点按order顺序连线
    def line(self, order, plant_length):
        all_time = 0
        plant_length += plant_graph[8][12]
        # 删除原线
        self.canvas.delete("line")
        order_1 = [12, 7, 11, 6, 10, 0, 1]
        order_2 = [2, 3, 9, 13, 8]
        order_3 = [8, 12, 7, 11, 6, 10, 0, 1, 5, 2, 3, 9, 13, 8]
        order_4 = [8, 7, 6, 4, 1, 5, 9, 8]
        for i in range(len(order_1) - 1):
            plant_length += plant_graph[order_1[i]][order_1[i + 1]]
        plant_length += plant_graph[2][5]
        for i in range(len(order_2) - 1):
            plant_length += plant_graph[order_2[i]][order_2[i + 1]]
        print("无人机的配送路线为：")
        print("9", end=" ")
        for i in range(len(order_1)):
            print(order_1[i] + 1, end=" ")
        print(",6", end=" ")
        for i in range(len(order_2)):
            print(order_2[i] + 1, end=" ")
        print("\n", end=" ")
        print("无人机的最短路径为：{}".format(plant_length))
        time_car = car_graph[order_4[0]][order_4[1]] / 50
        time_plant = 0
        time_plant = time_plant + ((plant_graph[order_3[0]][order_3[1]] + plant_graph[order_3[1]][
            order_3[2]]) / 75)
        all_time = all_time + max(time_car, time_plant)

        time_car = car_graph[order_4[1]][order_4[2]] / 50
        time_plant = 0
        time_plant = time_plant + ((plant_graph[order_3[2]][order_3[3]] + plant_graph[order_3[3]][
            order_3[4]]) / 75)
        all_time = all_time + max(time_car, time_plant)

        time_car = (car_graph[order_4[2]][order_4[3]] + car_graph[order_4[3]][order_4[4]]) / 50
        time_plant = 0
        time_plant = time_plant + ((plant_graph[order_3[4]][order_3[5]] + plant_graph[order_3[5]][
            order_3[6]] + plant_graph[order_3[6]][order_3[7]]) / 75)
        all_time = all_time + max(time_car, time_plant)

        all_time = all_time + (car_graph[1][4] + car_graph[4][8] + car_graph[8][5]) / 50

        time_car = car_graph[order_4[5]][order_4[6]] / 50
        time_plant = 0
        time_plant = time_plant + ((plant_graph[order_3[8]][order_3[9]] + plant_graph[order_3[9]][order_3[10]]) / 75)
        all_time = all_time + max(time_car, time_plant)

        time_car = car_graph[order_4[6]][order_4[7]] / 50
        time_plant = 0
        time_plant = time_plant + ((plant_graph[order_3[10]][order_3[11]] + plant_graph[order_3[11]][order_3[12]]) / 75)
        all_time = all_time + max(time_car, time_plant)
        print("一次整体配送所花时间最短为（小时）：{:.4f}".format(all_time))
        print("\n")
        y1.append(all_time)

        def line2(i1, i2):
            p1, p2 = self.nodes[i1], self.nodes[i2]
            if i2 in order:
                self.canvas.create_line(p1, p2, fill="black", tags="line")
            return i2

        def line3(i1, i2):
            p1, p2 = self.nodes[i1], self.nodes[i2]
            if i2 in order_1 or i2 in order_2:
                self.canvas.create_line(p1, p2, fill="blue", dash=(4, 4))
            return i2

        # order[-1]为初始值
        reduce(line2, order, order[-1])
        reduce(line3, order_1, 8)
        reduce(line3, order_2, 5)

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
            print(u"迭代次数：", self.iter, u"最佳路径总距离：", int(self.best_ant.total_distance))
            x1.append(self.iter)
            print("配送车辆的路线为：")
            for i in range(len(self.best_ant.path)):
                print(self.best_ant.path[i] + 1, end=" ")
            print(" ")
            print("配送车辆的最短路径为：{}".format(self.best_ant.total_distance))
            self.line(self.best_ant.path, plant_length)
            # 设置标题
            self.title("a:初始化 b:开始搜索 c:停止搜索 d:退出程序 迭代次数: %d" %
                       self.iter)
            # 更新画布
            self.canvas.update()
            self.iter += 1

    # 更新信息素
    def __update_pheromone_gragh(self):
        # 获取每只蚂蚁在其路径上留下的信息素
        temp_pheromone = [[0.0 for col in range(city_num)] for raw in range(city_num)]
        for ant in self.ants:
            for i in range(1, city_num - 6):
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


# ----------- 程序的入口处 -----------

xq = [12, 90, 24, 15, 70, 18, 150, 50, 30, 168, 36, 44, 42, 13]  # 需求
a = 1.1
D = 87.5
hc = []  # 货车停靠点
fj = []  # 可能的飞机停靠点
plant_in = [True for i in range(city_num)]  # 飞机是否可在此停靠
cl = [False for i in range(city_num)]  # 该结点是否以处理过
for i in range(city_num):
    if xq[i] > 50:
        plant_in[i] = False
        hc.append(i)
    else:
        fj.append(i)
fj.sort()
if len(hc) > 0:
    p1 = 1
else:
    p1 = fj[len(fj) - 1]
k = 0
s = []  # 蔟中的段
d_tkd = []  # 段中两个焦点（汽车停靠点）


def chuli(p1, p2):
    s.append([])
    a1 = [p1, p2]
    d_tkd.append(a1)
    cl[p1] = True
    cl[p2] = True
    for i in range(city_num):
        if plant_path[p1][i] + plant_path[p2][i] < a * D and plant_in[i] == True:
            if cl[i] == False:
                cl[i] = True
                s[k].append(i)
            else:
                for i1 in range(k):
                    n = len(s[i1])
                    j1 = 0
                    while j1 < n:
                        if s[i1][j1] == i:  # 如果这个被处理过的点离当前的两个汽车停靠点更近，则将点移到这里
                            if plant_path[d_tkd[i1][0]][i] + plant_path[d_tkd[i1][1]][i] > \
                                    plant_path[p1][i] + plant_path[p2][i]:
                                del s[i1][j1]
                                n = n - 1
                                s[k].append(i)
                        j1 = j1 + 1


def dian(a1, p1):
    c = 0
    for i in range(city_num):
        if p1 != i and plant_path[p1][i] <= a1 * D and cl[i] == False:
            c = c + 1
    return c


while 1:
    m1 = dian(1, p1)
    m = dian(a, p1)
    if m1 == 0:
        break
    if m == 1:
        for i in range(city_num):
            if p1 != i and plant_path[p1][i] <= a * D:
                a1 = [p1, i]
                d_tkd.append(a1)
                break
        break
    else:
        max1 = 0
        # 汽车停靠点集合
        for i in range(city_num):  # 如果没有汽车停放点，则选与p1构成椭圆包含未处理点更多的点
            if cl[i] == True or i == p1:
                continue
            cnt = 0
            qctkd = []
            for j in range(city_num):
                if plant_path[i][j] + plant_path[p1][j] <= a * D and cl[j] == False:
                    if xq[j] > 50:
                        qctkd.append(j)
                    else:
                        cnt += 1
            if cnt > max1:
                max1 = cnt
                p2 = i
            elif cnt == max1 and cnt != 0:
                if plant_path[p1][p2] > plant_path[p1][i]:
                    p2 = i
        if len(qctkd) > 0:  # 如果有多个汽车停放点，选最远的两个
            if xq[p1] > 50 or len(qctkd) > 1:
                max1 = 0
                qctkd.append(p1)
                for i in range(len(qctkd)):
                    if cl[i]:
                        continue
                    for j in range(len(qctkd)):
                        if cl[j]:
                            continue
                        if plant_path[i][j] > max1:
                            max1 = plant_path[i][j]
                            p1 = i
                            p2 = j
            else:
                p2 = qctkd[0]
    if car_graph[p1][p2] != distance_graph[p1][p2]:
        for i in range(city_num):
            if car_graph[p1][i] + car_graph[i][p2] == distance_graph[p1][p2] and cl[i] == False and i != p2:
                chuli(p1, i)
                k = k + 1
                chuli(i, p2)
                k = k + 1
                break
    else:
        chuli(p1, p2)
        k += 1
    p1 = p2

if __name__ == '__main__':
    TSP(tkinter.Tk()).mainloop()
