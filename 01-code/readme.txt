####################
case1: without detection
��Ӧ��python�Լ������epsͼƬ
ʵ��
python: sim_twohop_withoutdetect.py 
csv: without_detection_data.csv
��ͼ
plot_without_detection_data.m
eps: twohop_without_detection.eps


####################
case2: with fully detection
��Ӧ��python�Լ������epsͼƬ
ʵ��
python: sim_twohop_fulldetect.py 
csv: full_detection_data.csv
��ͼ
plot_full_detection_data.m
eps: twohop_with_fully_detection.eps

####################
�������(BVP���������)
ʵ�� analyze code:
analyse_twohop_MN.R
csv: opt_solve.csv
��ͼ
plot_twohop_MN_U.m
Ut.eps
state.eps

####################
ͼ�в���չʾ
simulation code: 
sim_twohop_MN_U.py

####################
cost v.s. (t_{on}, t_{off})
reward v.s. (t_{on}, t_{off})
ʵ�飺
sim_twohop_MN_U.py
���
result_***_time_seq_uniform_detect_M_N_U.tmp
result_***_cost_seq_uniform_detect_M_N_U.tmp
��ͼ
plot_surface_twohop_MN_U.m
cost_all_t0t1.eps
reward_all_t0t1.eps

####################
�Ƚ��㷨
cost v.s. alpha
ʵ��
analyse_twohop_MN_var_alpha.R
opt_solve_var_alpha.csv
sim_twohop_optimaldetect.py
matrix_cost_opt_wo_wc_half.csv
��ͼ
plot_matrix_cost.m
