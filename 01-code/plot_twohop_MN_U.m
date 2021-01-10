%% 表头  "x","1","2","U","I","S"\n  即 time, M(t), Lambda(t), U(t), I(t), S(t)
%% x, M, I, S, L, U
clear;
clc;
solve = csvread('opt_solve.csv', 1, 0);
time = solve(:,1);
D = solve(:,2);
I = solve(:,3);
R = solve(:,4);
L = solve(:,5);
U = solve(:,6);

%% plot U(t) control
figure(1)
Um = max(U);
% alpha = 0.9
plot(time,U,'color','blue','linestyle','-','Marker','s','MarkerIndices',1:250:length(U),'LineWidth',1);
hold on;
% alpha = 0.995 from *.R result
U2 = zeros(length(U),1);
plot(time,U2,'color','red','linestyle','--','Marker','o','MarkerIndices',1:250:length(U2),'LineWidth',1);
xlabel('Time (s)');
ylabel('U(t)');
ylim([0 Um*1.1])
t_on = annotation('textarrow', [0.35, 0.3], [0.3,0.16])
t_off = annotation('textarrow', [0.77, 0.82], [0.3,0.16]) 
set(t_on,'string','t_{on}', 'fontsize', 15)
set(t_off,'string','t_{off}', 'fontsize', 15)
set(gca, 'Fontname', 'Times New Roman','FontSize',15);
legend('\alpha=0.9','\alpha=0.995');

%% 两个结果？


%% plot M(t) I(t) S(t); lambda2（t） state
% 抽样250个点 画出一个点
figure(2)
plot(time, D,'color','blue','linestyle','-','Marker','s','MarkerIndices',1:250:length(D),'LineWidth',1);
hold on;
plot(time, I,'color','green','linestyle','-.','Marker','d','MarkerIndices',1:250:length(I),'LineWidth',1);
plot(time, R,'color','red','linestyle','-','Marker','^','MarkerIndices',1:250:length(R),'LineWidth',1);
plot(time, L,'color','magenta','linestyle','--','Marker','x','MarkerIndices',1:250:length(L),'LineWidth',1);
xlabel('Time (s)');
ylim([0 100*1.1])
legend('D(t)','I(t)','R(t)','\lambda_{D}(t)');
ylabel('Number of Nodes');
set(gca, 'Fontname', 'Times New Roman','FontSize',15);