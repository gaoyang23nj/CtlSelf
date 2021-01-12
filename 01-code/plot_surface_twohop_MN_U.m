clear;
clc;
all = csvread('result_20210110222121_time_seq_uniform_detect_M_N_U.tmp');
s = size(all);
%% t0:0,10,20,...490, // t1:1,11,21,...491
cnt = 0;
stepsize = 20;
T = 500;
%所有组合的(t0,t1),排除最优的一组
num_record = s(1,1)-1;
len_matrix = T/stepsize+1;
res = ones(len_matrix, len_matrix)*-1;
res_reward = ones(len_matrix, len_matrix)*-1;
for i = 1:num_record
    %数值
    t0 = all(i,1);
    t1 = all(i,2);
    %位置 0/20=0 放在位置1  500/20 = 25 放在位置26
    idx0 = floor(t0/stepsize)+1;
    idx1 = floor(t1/stepsize)+1;
    if res(idx0,idx1)==-1
        %t0,t1,(3)cost_reward,(4)cost_detection,(5)cost_rewardI,(6)total_cost,(7)total_reward
        res(idx0,idx1)=all(i,6);
        res_reward(idx0,idx1)=all(i,7);
        cnt = cnt + 1;
    else
        fprintf('Duplicate Err!!!\n')
    end
end
cnt

%% t0:0,10,20,...490, // repmat( A , m , n )
t0 = 0:stepsize:T;
t0 = t0';
t0 = repmat(t0,1,len_matrix);
% t1:1,11,21,...491， // 
t1 = 0:stepsize:T;
t1 = repmat(t1,len_matrix,1);

%% 存在一半的零值
for i=1:len_matrix
    for j=1:len_matrix
        if i>j
            res(i,j)= nan;
            res_reward(i,j)= nan;
        end
    end
end

%% figure(1) total_cost surf
figure(1)
surfc(t0,t1,res);
hold on;
ztmp = zlim;
plot3(all(end,1), all(end,2), ztmp(1), 'k*','MarkerSize',5);
text(all(end,1)-50, all(end,2)-50, ztmp(1)-20,  sprintf('(%.2f,%.2f,%.2f)',all(end,1), all(end,2), all(end,6)))
xlabel('t_{on}');
ylabel('t_{off}');
zlabel('cost');
colorbar('position',[0.8 0.3 0.025 0.45])
set(gca, 'Fontname', 'Times New Roman','FontSize',15);


%% figure(2) reward
figure(2)
surfc(t0,t1,res_reward);
hold on;
ztmp = zlim;
plot3(all(end,1), all(end,2), ztmp(1), 'k*','MarkerSize',5);
text(all(end,1)-50, all(end,2)-50, ztmp(1)-20,  sprintf('(%.2f,%.2f,%.2f)',all(end,1), all(end,2), all(end,7)))
xlabel('t_{on}');
ylabel('t_{off}');
zlabel('reward');
colorbar('position',[0.8 0.3 0.025 0.45])
set(gca, 'Fontname', 'Times New Roman','FontSize',15);

p = all(end,3)/(all(end,3) + all(end,5))
