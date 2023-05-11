####################
case1: without detection
对应的python以及输出的eps图片
实验
python: sim_twohop_withoutdetect.py 
csv: without_detection_data.csv
画图
plot_without_detection_data.m
eps: twohop_without_detection.eps


####################
case2: with fully detection
对应的python以及输出的eps图片
实验
python: sim_twohop_fulldetect.py 
csv: full_detection_data.csv
画图
plot_full_detection_data.m
eps: twohop_with_fully_detection.eps

####################
最优求解(BVP问题求解器)
实验 analyze code:
analyse_twohop_MN.R
csv: opt_solve.csv
画图
plot_twohop_MN_U.m
Ut.eps
state.eps

####################
图中并无展示
simulation code: 
sim_twohop_MN_U.py

####################
cost v.s. (t_{on}, t_{off})
reward v.s. (t_{on}, t_{off})
实验：
sim_twohop_MN_U.py
输出
result_***_time_seq_uniform_detect_M_N_U.tmp
result_***_cost_seq_uniform_detect_M_N_U.tmp
画图
plot_surface_twohop_MN_U.m
cost_all_t0t1.eps
reward_all_t0t1.eps

####################
比较算法
cost v.s. alpha
实验
analyse_twohop_MN_var_alpha.R
opt_solve_var_alpha.csv
sim_twohop_optimaldetect.py
matrix_cost_opt_wo_wc_half.csv
画图
plot_matrix_cost.m
