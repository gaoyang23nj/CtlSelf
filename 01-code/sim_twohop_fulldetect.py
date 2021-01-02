import datetime
import numpy as np
import matplotlib.pyplot as plt

para_N = 100
para_total_time = 2500
# 指数生成
para_lambda = 0.004
para_rho = 0.01

para_alpha = 0.9
# 检查强度 每秒执行多少次检查
para_Um = 2

event_fromsrc = 'rcv_from_src'
event_selfish = 'to_be_selfish'
event_detect = 'detect'

state_without_message = 0
state_with_message = 1
state_selfish = 2

# selfish的产生 用接触式模拟
# src独立于N个nodes，不被感染
class Sim(object):
    def __init__(self, num_nodes, total_time):
        self.total_sim_time = total_time
        self.N = num_nodes
        # 当前运行时间
        self.running_time = 0
        # 总体上下一次相遇事件的时刻
        self.list_nextContact = []
        # 假定的selfish源头
        self.sel_nextContact = np.ones(self.N, dtype='float') * -1
        # 假定message源头
        self.src_nextContact = np.ones(self.N, dtype='float') * -1

        # 假定检查时间
        self.detect_nextContact = 0.

        # 每个node的状态 -- 矩阵
        # 0 表示 without_message; 1 表示with_message; 2 表示selfish状态
        self.stateNode = np.zeros(self.N, dtype='int')

        # 初始化next_contact_time矩阵
        self.__init_contact_time()

        # print(self.list_nextContact)
        # print(self.nextContact)

        # 检测能力的使用
        self.res_detect_ability = []
        # 中间结果
        self.res_record = []
        # 保存结果
        self.time_index = np.arange(0, self.total_sim_time)
        self.res_nr_nodes_no_message = np.ones(self.total_sim_time, dtype='int') * -1
        self.res_nr_nodes_with_message = np.ones(self.total_sim_time, dtype='int') * -1
        self.res_nr_nodes_selfish = np.ones(self.total_sim_time, dtype='int') * -1
        (t, nr_h, nr_i, nr_m) = self.update_nr_nodes_record_with_time()
        self.res_nr_nodes_no_message[0] = nr_h
        self.res_nr_nodes_with_message[0] = nr_i
        self.res_nr_nodes_selfish[0] = nr_m
        while True:
            # next_time > total_sim
            if self.list_nextContact[0][0] >= self.total_sim_time:
                break
            # 相遇投递 新投递时间更新 selfish变异
            self.run()

        # 整理结果
        for i in range(1, self.total_sim_time):
            is_found = False
            for j in range(len(self.res_record)):
                (t, nr_h, nr_i, nr_m) = self.res_record[j]
                if t > i:
                    break
                if t <= i:
                    is_found = True
                    self.res_nr_nodes_no_message[i] = nr_h
                    self.res_nr_nodes_with_message[i] = nr_i
                    self.res_nr_nodes_selfish[i] = nr_m
            if not is_found:
                self.res_nr_nodes_no_message[i] = self.res_nr_nodes_no_message[i - 1]
                self.res_nr_nodes_with_message[i] = self.res_nr_nodes_with_message[i - 1]
                self.res_nr_nodes_selfish[i] = self.res_nr_nodes_selfish[i - 1]

    @staticmethod
    def get_next_wd(pl):
        res = np.random.exponential(1 / pl)
        return res

    # cost 计算
    def get_cost(self, granularity=1):
        cost_from_sel = np.sum(self.res_nr_nodes_selfish) * granularity
        cost_from_detect = len(self.res_detect_ability)
        return cost_from_sel, cost_from_detect

    # 损失 多少% 的 reward
    def get_sim_res(self):
        reward_for_with = np.sum(self.res_nr_nodes_with_message)
        reward_for_selfish = np.sum(self.res_nr_nodes_selfish)
        p = reward_for_selfish / (reward_for_with + reward_for_selfish)
        return p, self.time_index, self.res_nr_nodes_no_message, self.res_nr_nodes_with_message, self.res_nr_nodes_selfish

    def update_nr_nodes_record_with_time(self):
        nr_h, target_list = self.get_sel_state_node(state=0)
        # self.res_nr_nodes_no_message[time_idx] = nr_h

        nr_i, target_list = self.get_sel_state_node(state=1)
        # self.res_nr_nodes_with_message[time_idx] = nr_i

        nr_m, target_list = self.get_sel_state_node(state=2)
        # self.res_nr_nodes_selfish[time_idx] = nr_m

        return self.running_time, nr_h, nr_i, nr_m

    def get_sel_state_node(self, state):
        tmp_i = np.sum(self.stateNode == state)
        # 获取对应的 index_list
        index_list = np.argwhere(self.stateNode == state)
        index_list = np.squeeze(index_list, axis=1)
        index_list = index_list.tolist()
        # 选择一个节点(选第一个) // 现在暂时不考虑 src不变异的假设
        np.random.shuffle(index_list)
        return tmp_i, index_list

    def __init_contact_time(self):
        # 节点可能变异的事件; to be selfish; 0时刻 不必执行 从1时刻开始
        for i in range(self.N):
            self.sel_nextContact[i] = self.get_next_wd(para_rho)
            self.list_nextContact.append((self.sel_nextContact[i], event_selfish, i))

        # 节点收到message
        for i in range(self.N):
            self.src_nextContact[i] = self.get_next_wd(para_lambda)
            self.list_nextContact.append((self.src_nextContact[i], event_fromsrc, i))

        # 进行检查/抽查
        # 以何种方式执行？ 匀速/
        T_m = 1/para_Um
        tmp = 0
        end_detection = self.total_sim_time
        while i <= end_detection:
            tmp = tmp + para_Um
            if tmp >= 1:
                self.detect_nextContact = i
                self.list_nextContact.append((self.detect_nextContact, event_detect))
                # tmp_print_list_detect.append((self.detect_nextContact, event_detect))
                tmp = tmp - 1
            i = i + 1
        # print(tmp_print_list_detect)

        self.list_nextContact.sort()

    def run(self):
        if self.list_nextContact[0][1] == event_detect:
            # print(self.list_nextContact[0])
            # print(self.running_time, self.list_nextContact)

            (t, eve) = self.list_nextContact[0]
            assert self.running_time <= t
            # 更新新的时间 和 pop 事件
            self.running_time = t
            self.list_nextContact.pop(0)

            target_detect = np.random.randint(0, self.N)
            if self.stateNode[target_detect] == state_selfish:
                self.stateNode[target_detect] = state_without_message
            # self.update_to_detect()
            self.res_detect_ability.append(self.running_time)
            self.res_record.append(self.update_nr_nodes_record_with_time())
        elif self.list_nextContact[0][1] == event_fromsrc:
            # print(self.list_nextContact[0])
            # print(self.running_time, self.list_nextContact)

            (t, eve, i) = self.list_nextContact[0]
            assert self.running_time <= t
            # 更新新的时间 和 pop 事件
            self.running_time = t
            self.list_nextContact.pop(0)

            if self.stateNode[i] != state_with_message:
                self.stateNode[i] = state_with_message
            self.update_to_from_src(i)
            self.res_record.append(self.update_nr_nodes_record_with_time())
        elif self.list_nextContact[0][1] == event_selfish:
            # print(self.list_nextContact[0])
            # print(self.running_time, self.list_nextContact)

            (t, eve, i) = self.list_nextContact[0]
            assert self.running_time <= t
            # 更新新的时间 和 pop 事件
            self.running_time = t
            self.list_nextContact.pop(0)

            self.update_to_be_selfish(i)
            if self.stateNode[i] == state_with_message:
                self.stateNode[i] = state_selfish
            self.res_record.append(self.update_nr_nodes_record_with_time())
        else:
            print('Internal Err! -- unkown event time:{} eve_list:{}'.format(self.running_time, self.list_nextContact))

    def update_to_from_src(self, i):
        tmp_next_time = self.get_next_wd(para_lambda) + self.running_time
        self.src_nextContact[i] = tmp_next_time
        loc = 0
        for loc in range(len(self.list_nextContact)):
            if self.list_nextContact[loc][0] >= tmp_next_time:
                break
        self.list_nextContact.insert(loc, (tmp_next_time, event_fromsrc, i))

    def update_to_be_selfish(self, i):
        tmp_next_time = self.get_next_wd(para_rho) + self.running_time
        self.sel_nextContact[i] = tmp_next_time
        loc = 0
        for loc in range(len(self.list_nextContact)):
            if self.list_nextContact[loc][0] >= tmp_next_time:
                break
        self.list_nextContact.insert(loc, (tmp_next_time, event_selfish, i))


def try_once(is_plot):
    total_time = para_total_time
    the = Sim(para_N, total_time)
    p, x, r, i, d = the.get_sim_res()
    cost1, cost2 = the.get_cost()
    print('lost ({}) reward'.format(p))
    print('cost_from_sel:{} cost_from_detect:{}'.format(cost1, cost2))
    _ = plt.plot(x, r, label="R", color='green', marker='o', markersize=0.5)
    _ = plt.plot(x, i, label="I", color='blue', marker='o', markersize=0.5)
    _ = plt.plot(x, d, label="D", color='red', marker='o', markersize=0.5)

    if is_plot:
        # Show the plot
        draw_bound(total_time)
        draw_predict(total_time)

        _ = plt.xlabel('total waiting time (games)')
        _ = plt.ylabel('H, I, M')
        plt.legend()
        # Show the plot
        plt.show()


def try_10_times(run_times, is_plot):
    total_time = para_total_time
    # 各种状态
    multi_times_r = np.zeros((run_times, total_time))
    multi_times_i = np.zeros((run_times, total_time))
    multi_times_d = np.zeros((run_times, total_time))
    # percent of wasted reward
    multi_times_p = np.zeros(run_times)
    # cost from D(t); cost from U(t)
    multi_times_D = np.zeros(run_times)
    multi_times_U = np.zeros(run_times)
    for k in range(run_times):
        # print('*'*20, k)
        the = Sim(para_N, total_time)
        p, x, r, i, d = the.get_sim_res()
        cost_of_selfish, cost_of_detection = the.get_cost()
        # print('lost ({}) reward'.format(p))
        # print('cost_from_sel:{} cost_from_detect:{}'.format(cost1, cost2))
        multi_times_r[k, :] = r
        multi_times_i[k, :] = i
        multi_times_d[k, :] = d
        multi_times_p[k] = p
        multi_times_D[k] = cost_of_selfish
        multi_times_U[k] = cost_of_detection

    r = np.sum(multi_times_r, axis=0) / run_times
    i = np.sum(multi_times_i, axis=0) / run_times
    d = np.sum(multi_times_d, axis=0) / run_times
    cost_of_selfish = np.sum(multi_times_D) / run_times
    cost_of_detection = np.sum(multi_times_U) / run_times
    p = np.sum(multi_times_p) / run_times
    print('lost ({}) reward'.format(p))

    total_cost = cost_of_selfish * (1-para_alpha) + cost_of_detection * para_alpha
    print('cost_from_sel:{} cost_from_detect:{} total_cost:{}'.format(cost_of_selfish, cost_of_detection, total_cost))
    new_cost = cost_of_selfish/100*0.5 + cost_of_detection * 0.5
    print('new_cost:{}'.format(new_cost))
    print('\n')

    if is_plot:
        _ = plt.plot(x, r, label="r", color='green', linestyle='-')
        _ = plt.plot(x, i, label="i", color='blue', linestyle='-')
        _ = plt.plot(x, d, label="d", color='red', linestyle='-')

        # draw_bound(total_time)
        draw_predict(total_time)

        _ = plt.xlabel('total waiting time (games)')
        _ = plt.ylabel('R, I, D')
        plt.legend()
        # Show the plot
        plt.show()
    return cost_of_selfish, cost_of_detection, total_cost, new_cost


def draw_predict(total_time):
    x = np.arange(0, total_time)

    def func_nr_i(t):
        para_ele = -(para_lambda + para_rho) * t
        para_fraction = (para_lambda * para_N) / (para_lambda + para_rho)
        result = para_fraction * (1 - np.math.exp(para_ele))
        return result

    def func_nr_d(t):
        para_ele1 = -(para_lambda + para_Um/para_N) * t
        para_ele2 = -(para_lambda + para_rho) * t
        para_fraction1 = (para_rho * para_lambda * para_N) / ((para_lambda + para_Um/para_N)*(para_Um/para_N - para_rho))
        para_fraction2 = (para_rho * para_lambda * para_N) / ((para_lambda + para_rho)*(para_Um/para_N - para_rho))
        para_plus = (para_rho * para_lambda * para_N) / ((para_lambda + para_rho)*(para_lambda + para_Um/para_N))
        result = - para_fraction1 * np.math.exp(para_ele1) + para_fraction2 * np.math.exp(para_ele2) + para_plus
        return result

    sim_i = np.ones(total_time) * -1
    sim_d = np.ones(total_time) * -1
    for i in range(total_time):
        sim_i[i] = func_nr_i(i)
        sim_d[i] = func_nr_d(i)
    _ = plt.plot(x, sim_i, label="predict_I", color='black', linestyle='--')
    _ = plt.plot(x, sim_d, label="predict_D", color='yellow', linestyle='--')

    # 比例
    prop = np.true_divide(sim_d, sim_i+0.000001)
    # _ = plt.plot(x, prop, label="prop", color='black', linestyle='--')


def draw_bound(total_time):
    x = np.arange(0, total_time)
    para_i = (para_lambda * para_N) / (para_lambda + para_rho)
    para_d = (para_rho * para_N) / (para_lambda + para_rho)
    bound_i = np.ones(total_time) * para_i
    bound_d = np.ones(total_time) * para_d
    _ = plt.plot(x, bound_i, label="bound_i", color='blue', linestyle='--')
    _ = plt.plot(x, bound_d, label="bound_m", color='red', linestyle='--')


if __name__ == "__main__":
    t1 = datetime.datetime.now()

    # try_once(True)

    # # 优化结果
    try_10_times(20, True)

    t2 = datetime.datetime.now()
    print(t2-t1)

