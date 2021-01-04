%% ±íÍ·  "x","1","2","U","I","S"\n  ¼´ time, M(t), Lambda(t), U(t), I(t), S(t)
solve = csvread('opt_solve.csv', 1, 0);
time = solve(:,1);
M = solve(:,2);
Lambda2 = solve(:,3);
U = solve(:,4);
I = solve(:,5);
S = solve(:,6);

%% plot U(t) control
figure(1)
Um = max(U);
plot(time,U,'o','markersize',1.5,'color','b','linestyle','-');
xlabel('time');
ylabel('U(t)');
ylim([0 Um*1.1])
set(gca, 'Fontname', 'Times New Roman','FontSize',15);

%% plot M(t) I(t) S(t); lambda2£¨t£© state
% plot(time, M,'o','markersize',1.5,'color','k','linestyle','-');
% hold on;
% plot(time, I,'^','markersize',1.5,'color','r','linestyle','-');
% plot(time, S,'*','markersize',1.5,'color','y','linestyle','-');
% plot(time, Lambda2,'s','markersize',1.5,'color','b','linestyle','--');
figure(2)
plot(time, M,'color','k','linestyle','-');
hold on;
plot(time, I,'color','r','linestyle','-');
plot(time, S,'color','y','linestyle','-');
plot(time, Lambda2,'color','b','linestyle','--');
xlabel('time');
ylim([0 100*1.1])
legend('M(t)','I(t)','S(t)','\lambda_{2}(t)')
%ylabel('U(t)');
set(gca, 'Fontname', 'Times New Roman','FontSize',15);