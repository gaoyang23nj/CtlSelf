clear;
clc;
all = csvread('matrix_cost_opt_wo_wc_half.csv');
%all = all(1:end-1,:);
alpha = all(:,1);
opt = all(:,2);
wo = all(:,3);
wc = all(:,4);
w_half = all(:,5);
plot(alpha, opt, 'color','red','linestyle','-','LineWidth',2, 'Marker','^','MarkerIndices',1:3:length(opt), 'MarkerSize',8);
hold on;
plot(alpha, wo, 'color','black','linestyle','-.','LineWidth',2, 'Marker','s','MarkerIndices',1:3:length(wo), 'MarkerSize',8);
plot(alpha, wc, 'color','blue','linestyle','--','LineWidth',2, 'Marker','o','MarkerIndices',1:3:length(wc), 'MarkerSize',8);
plot(alpha, w_half, 'color','green','linestyle','--','LineWidth',2, 'Marker','d','MarkerIndices',1:3:length(w_half), 'MarkerSize',8);
legend('our method','without detection','with complete detection', 'with half detection');
ylabel('cost','FontSize',12)
xlabel('\alpha','FontSize',12)
% grid on;
x_min = min(all(:,1));
x_max = max(all(:,1));
xlim([x_min, x_max]);
% ylim([0, 100])
set(gca, 'Fontname', 'Times New Roman','FontSize',15);
set(gcf,'position',[400,100,880,660]);