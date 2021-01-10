# [Read me]
# 所有(t0,t1)情况 + optimal solution的遍历数据
# 输出到 '********_uniform_detect_M_N_U.tmp'

import datetime
import numpy as np

para_N = 100
para_total_time = 500
# 指数生成
para_lambda = 0.004
para_rho = 0.011

para_alpha = 0.9
# 检查强度 每秒执行多少次检查
para_Um = 1
# per reward unit  = 0.1
para_beta = 0.1

para_time_granularity = 0.1

event_fromsrc = 'rcv_from_src'
event_selfish = 'to_be_selfish'
event_detect = 'detect'

state_without_message = 0
state_with_message = 1
state_selfish = 2


# selfish的产生 用接触式模拟
# src独立于N个nodes，不被感染
class Sim(object):
    def __init__(self, num_nodes, total_time, t0, t1):
        # 最小的检测时间
        self.t0 = t0
        # 最大检测时间
        self.t1 = t1
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
        # 保存结果 0.1->10 1个单位事件对应多少个granularity
        self.per = int(1/para_time_granularity)
        self.time_index = np.arange(0, self.total_sim_time*self.per)
        self.res_nr_nodes_no_message = np.ones(self.total_sim_time*self.per, dtype='int') * -1
        self.res_nr_nodes_with_message = np.ones(self.total_sim_time*self.per, dtype='int') * -1
        self.res_nr_nodes_selfish = np.ones(self.total_sim_time*self.per, dtype='int') * -1
        # 为了便于仿真统计 初始时刻的状态 需要记录一下
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
        # for i in range(1, self.total_sim_time):time
        i = 1
        time = 0 + para_time_granularity
        while time <= self.total_sim_time:
            is_found = False
            for j in range(len(self.res_record)):
                (t, nr_h, nr_i, nr_m) = self.res_record[j]
                if t > time:
                    break
                # if t <= time:
                # 对于之前的 可以不停的复制 直到停止
                else:
                    is_found = True
                    self.res_nr_nodes_no_message[i] = nr_h
                    self.res_nr_nodes_with_message[i] = nr_i
                    self.res_nr_nodes_selfish[i] = nr_m
            # 如果这个时间间隔没有新的事件发生 就是用之前的统计状态
            if not is_found:
                self.res_nr_nodes_no_message[i] = self.res_nr_nodes_no_message[i - 1]
                self.res_nr_nodes_with_message[i] = self.res_nr_nodes_with_message[i - 1]
                self.res_nr_nodes_selfish[i] = self.res_nr_nodes_selfish[i - 1]
            time = time + para_time_granularity
            i = i+1


    @staticmethod
    def get_next_wd(pl):
        res = np.random.exponential(1 / pl)
        return res

    # cost 计算
    def get_cost(self, granularity=para_time_granularity):
        cost_from_sel = np.sum(self.res_nr_nodes_selfish) * granularity
        # cost_from_detect = len(self.res_detect_ability)
        cost_from_detect = self.t1 - self.t0
        cost_from_rewardI = np.sum(self.res_nr_nodes_with_message) * granularity
        return cost_from_sel, cost_from_detect, cost_from_rewardI

    # 损失 多少% 的 reward
    def get_sim_res(self):
        reward_for_with = np.sum(self.res_nr_nodes_with_message)
        reward_for_selfish = np.sum(self.res_nr_nodes_selfish)
        p = reward_for_selfish / (reward_for_with + reward_for_selfish)
        return p, self.time_index, self.res_nr_nodes_no_message, self.res_nr_nodes_with_message, \
               self.res_nr_nodes_selfish

    def update_nr_nodes_record_with_time(self):
        nr_h, target_list = self.get_sel_state_node(state=state_without_message)
        # self.res_nr_nodes_no_message[time_idx] = nr_h

        nr_i, target_list = self.get_sel_state_node(state=state_with_message)
        # self.res_nr_nodes_with_message[time_idx] = nr_i

        nr_m, target_list = self.get_sel_state_node(state=state_selfish)
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
        # 以何种方式执行？
        T_m = 1/para_Um
        end_detection = self.t1
        if self.total_sim_time < self.t1:
            end_detection = self.total_sim_time
        tmp_now = self.t0 + T_m
        while tmp_now <= end_detection:
            self.list_nextContact.append((tmp_now, event_detect))
            tmp_now = tmp_now + T_m

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


def try_10_times(t0, t1, run_times):
    total_time = para_total_time
    # 各种状态
    per = int(1 / para_time_granularity)
    multi_times_r = np.zeros((run_times, total_time*per))
    multi_times_i = np.zeros((run_times, total_time*per))
    multi_times_d = np.zeros((run_times, total_time*per))
    # percent of wasted reward
    multi_times_p = np.zeros(run_times)
    # cost from D(t); cost from U(t)
    multi_times_D = np.zeros(run_times)
    multi_times_U = np.zeros(run_times)
    multi_times_I = np.zeros(run_times)
    for k in range(run_times):
        # print('*'*20, k)
        the = Sim(para_N, total_time, t0, t1)
        p, x, r, i, d = the.get_sim_res()
        cost_of_selfish, cost_of_detection, cost_of_rewardI = the.get_cost()
        # print('lost ({}) reward'.format(p))
        # print('cost_from_sel:{} cost_from_detect:{}'.format(cost1, cost2))
        multi_times_r[k, :] = r
        multi_times_i[k, :] = i
        multi_times_d[k, :] = d
        multi_times_p[k] = p
        multi_times_D[k] = cost_of_selfish
        multi_times_U[k] = cost_of_detection
        multi_times_I[k] = cost_of_rewardI
    r = np.sum(multi_times_r, axis=0) / run_times
    i = np.sum(multi_times_i, axis=0) / run_times
    d = np.sum(multi_times_d, axis=0) / run_times
    cost_of_selfish = np.sum(multi_times_D) / run_times
    cost_of_detection = np.sum(multi_times_U) / run_times
    cost_of_rewardI = np.sum(multi_times_I) / run_times
    p = np.sum(multi_times_p) / run_times
    print('*** avg *** t0:{} t1:{}'.format(t0, t1))
    print('lost ({}) reward'.format(p))

    total_cost = cost_of_selfish * (1-para_alpha) + cost_of_detection * para_alpha
    print('cost_from_sel:{} cost_from_detect:{} total_cost:{}'.format(cost_of_selfish, cost_of_detection, total_cost))
    # new_cost = cost_of_selfish/100*0.5 + cost_of_detection * 0.5
    # print('new_cost:{}'.format(new_cost))
    total_reward = (cost_of_rewardI + cost_of_selfish) * para_beta
    print('total_reward:{}'.format(total_reward))
    print('\n')
    return cost_of_selfish, cost_of_detection, cost_of_rewardI, total_cost, total_reward


def draw_all_combine(total_time, delta):
    list_t0_t1 = []
    # total_time+1 就能遍历到最后一个
    for i in range(0, total_time+delta, delta):
        for j in range(i, total_time+delta, delta):
            list_t0_t1.append((i, j))
    print(list_t0_t1)
    # 保存结果
    res = []
    for (t0, t1) in list_t0_t1:
        c1, c2, c3, tc, tr = try_10_times(t0, t1, 20)
        ele = (t0, t1, c1, c2, c3, tc, tr, total_time, delta)
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
    delta = 20
    # t0-t1遍历结果
    res = draw_all_combine(para_total_time, delta)

    # try_once(0, 5000, True)
    # 优化结果
    o_t0 = 102.1
    o_t1 = 452.3
    c1, c2, c3, tc, tr = try_10_times(o_t0, o_t1, 20)
    ele = (o_t0, o_t1, c1, c2, c3, tc, tr, para_total_time, 0)
    res.append(ele)

    short_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    print_file(res, short_time+'_time_seq')

    def take_total_cost(elem):
        return elem[5]

    res.sort(key=take_total_cost)
    print_file(res, short_time+'_cost_seq')
    t2 = datetime.datetime.now()
    print(t2-t1)

