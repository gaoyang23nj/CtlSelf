import datetime
import numpy as np
import matplotlib.pyplot as plt

para_N = 100
para_total_time = 500
# 指数生成
para_lambda = 0.004
para_rho = 0.01

para_alpha = 0.9
para_Um = 1

event_fromsrc = 'rcv_from_src'
# event_contact = 'contact'
event_selfish = 'to_be_selfish'
event_detect = 'detect'


# selfish的产生 用接触式模拟
# src独立于N个nodes，不被感染
class Sim(object):
    def __init__(self, num_nodes, total_time, t0, t1):
        # 最小的检测时间
        self.t0 = t0
        # 最大检测时间
        self.t1 = t1
        # 检查强度 每秒执行多少次检查
        self.U_m = 5
        self.total_sim_time = total_time
        self.N = num_nodes
        # 当前运行时间
        self.running_time = 0
        # 每个pair下一次相遇的时刻 -- 矩阵
        self.nextContact = np.ones((self.N, self.N), dtype='float')*-1
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
        # 初始化下一次相遇的时刻; 并加入 时间list
        # for i in range(self.N):
        #     for j in range(self.N):
        #         # 只需要一半
        #         if i >= j:
        #             continue
        #         else:
        #             self.nextContact[i, j] = self.get_next_wd(para_lambda)
        #             self.list_nextContact.append((self.nextContact[i, j], event_contact, i, j))

        # 节点可能变异的事件; to be selfish; 0时刻 不必执行 从1时刻开始
        for i in range(self.N):
            self.sel_nextContact[i] = self.get_next_wd(para_rho)
            self.list_nextContact.append((self.sel_nextContact[i], event_selfish, i))

        # 节点收到message
        for i in range(self.N):
            self.src_nextContact[i] = self.get_next_wd(para_lambda)
            self.list_nextContact.append((self.src_nextContact[i], event_fromsrc, i))

        # 进行检查/抽查
        # 以何种方式执行？ 匀速/Poisson分布/随机？
        # tmp_print_list_detect = []
        tmp = 0
        i = self.t0
        while i <= self.t1:
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
            if self.stateNode[target_detect] == 2:
                self.stateNode[target_detect] = 0
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

            if self.stateNode[i] != 1:
                self.stateNode[i] = 1
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
            if self.stateNode[i] == 1:
                self.stateNode[i] = 2
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

    def update_to_detect(self):
        tmp_next_time = self.get_next_wd(para_Um) + self.running_time
        # 根据控制策略 停止检测
        if tmp_next_time > self.t1:
            return
        self.detect_nextContact = tmp_next_time
        loc = 0
        for loc in range(len(self.list_nextContact)):
            if self.list_nextContact[loc][0] >= tmp_next_time:
                break
        self.list_nextContact.insert(loc, (tmp_next_time, event_detect))

def try_once(t0, t1, is_plot):
    total_time = para_total_time
    the = Sim(para_N, total_time, t0, t1)
    p, x, h, i, m = the.get_sim_res()
    cost1, cost2 = the.get_cost()
    print('lost ({}) reward'.format(p))
    print('cost_from_sel:{} cost_from_detect:{}'.format(cost1, cost2))
    _ = plt.plot(x, h, label="h", color='green', marker='o', markersize=0.5)
    _ = plt.plot(x, i, label="i", color='blue', marker='o', markersize=0.5)
    _ = plt.plot(x, m, label="m", color='red', marker='o', markersize=0.5)

    if is_plot:
        # Show the plot
        draw_bound(total_time)
        draw_predict(total_time)

        _ = plt.xlabel('total waiting time (games)')
        _ = plt.ylabel('H, I, M')
        plt.legend()
        # Show the plot
        plt.show()


def try_10_times(t0, t1, is_plot):
    total_time = para_total_time
    run_times = 20
    t_h = np.zeros((run_times, total_time))
    t_i = np.zeros((run_times, total_time))
    t_m = np.zeros((run_times, total_time))
    t_p = np.zeros(run_times)
    t_cost1 = np.zeros(run_times)
    t_cost2 = np.zeros(run_times)
    for k in range(run_times):
        # print('*'*20, k)
        the = Sim(para_N, total_time, t0, t1)
        p, x, h, i, m = the.get_sim_res()
        cost1, cost2 = the.get_cost()
        # print('lost ({}) reward'.format(p))
        # print('cost_from_sel:{} cost_from_detect:{}'.format(cost1, cost2))
        t_cost1[k] = cost1
        t_cost2[k] = cost2
        t_p[k] = p
        t_h[k, :] = h
        t_i[k, :] = i
        t_m[k, :] = m
    cost1 = np.sum(t_cost1) / run_times
    cost2 = np.sum(t_cost2) / run_times
    p = np.sum(t_p) / run_times
    h = np.sum(t_h, axis=0) / run_times
    i = np.sum(t_i, axis=0) / run_times
    m = np.sum(t_m, axis=0) / run_times
    print('*** avg *** t0:{} t1:{}'.format(t0, t1))
    print('lost ({}) reward'.format(p))

    total_cost = cost1 * (1-para_alpha) + cost2 * para_alpha
    print('cost_from_sel:{} cost_from_detect:{} total_cost:{}'.format(cost1, cost2, total_cost))
    new_cost = cost1/100*0.5 + cost2 * 0.5
    print('new_cost:{}'.format(new_cost))
    print('\n')

    if is_plot:
        _ = plt.plot(x, h, label="h", color='green', linestyle='-')
        _ = plt.plot(x, i, label="i", color='blue', linestyle='-')
        _ = plt.plot(x, m, label="m", color='red', linestyle='-')

        draw_bound(total_time)
        draw_predict(total_time)

        _ = plt.xlabel('total waiting time (games)')
        _ = plt.ylabel('H, I, M')
        plt.legend()
        # Show the plot
        plt.show()
    return cost1, cost2, total_cost, new_cost


def draw_predict(total_time):
    x = np.arange(0, total_time)

    def func_nr_i(t):
        para_ele = -(para_lambda + para_rho) * t
        para_frac = (para_lambda * para_N) / (para_lambda + para_rho)
        res = para_frac * (1 - np.math.exp(para_ele))
        return res

    def func_nr_m(t):
        para_ele = -(para_lambda + para_rho) * t
        para_frac = (para_lambda * para_N) / (para_lambda + para_rho)
        para_plus = (para_rho * para_N) / (para_lambda + para_rho)
        res = - para_N * np.math.exp(- para_lambda * t) + para_frac * np.math.exp(para_ele) + para_plus
        return res

    sim_i = np.ones(total_time) * -1
    sim_m = np.ones(total_time) * -1
    for i in range(total_time):
        sim_i[i] = func_nr_i(i)
        sim_m[i] = func_nr_m(i)
    _ = plt.plot(x, sim_i, label="predict_i", color='black', linestyle='--')
    _ = plt.plot(x, sim_m, label="predict_m", color='yellow', linestyle='--')

    # 比例
    prop = np.true_divide(sim_m, sim_i+0.000001)
    _ = plt.plot(x, prop, label="prop", color='black', linestyle='--')


def draw_bound(total_time):
    x = np.arange(0, total_time)
    para_frac = (para_lambda * para_N) / (para_lambda + para_rho)
    para_plus = (para_rho * para_N) / (para_lambda + para_rho)
    pred_i = np.ones(total_time) * para_frac
    pred_m = np.ones(total_time) * para_plus
    _ = plt.plot(x, pred_i, label="bound_i", color='blue', linestyle='--')
    _ = plt.plot(x, pred_m, label="bound_m", color='red', linestyle='--')


def draw_all_combine(total_time, delta):
    list_t0_t1 = []
    for i in range(0, total_time+1, delta):
        for j in range(i+delta, total_time+1, delta):
            list_t0_t1.append((i, j))
    print(list_t0_t1)
    # 保存结果
    res = []
    for (t0, t1) in list_t0_t1:
        c1, c2, tc, nc = try_10_times(t0, t1, False)
        ele = (t0, t1, c1, c2, tc, nc, total_time, delta)
        res.append(ele)
    return res


def print_file(res, short_time):
    file_object = open('result_'+short_time+'_uniform_detect_M_N_U.tmp', 'a+', encoding="utf-8")
    for ele in res:
        tmp_string = ''
        for i in range(len(ele)):
            tmp_string = tmp_string + str(ele[i])
            if i != len(ele)-1:
                tmp_string = tmp_string + ','
        file_object.write(tmp_string + '\n')
    file_object.close()


if __name__ == "__main__":
    t1 = datetime.datetime.now()
    res = draw_all_combine(para_total_time, 10)

    # try_once(0, 5000, True)
    # 优化结果
    o_t0 = 108+1
    o_t1 = 451
    c1, c2, tc, nc = try_10_times(o_t0, o_t1, False)
    ele = (o_t0, o_t1, c1, c2, tc, nc, para_total_time, 0)
    res.append(ele)

    short_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    print_file(res, short_time+'_time_seq')

    def take5(elem):
        return elem[4]

    res.sort(key=take5)
    print_file(res, short_time+'_cost_seq')
    t2 = datetime.datetime.now()
    print(t2-t1)

