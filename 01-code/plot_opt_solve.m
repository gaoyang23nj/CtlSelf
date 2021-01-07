clear all;
load('opt_solve.mat');
a=table2array(optsolve);
Len_x=size(optsolve,2);
Len_y=size(optsolve,1);
re_1=zeros(1,(Len_y-1)/20);
re_2=zeros(1,(Len_y-1)/20);
re_3=zeros(1,(Len_y-1)/20);
re_4=zeros(1,(Len_y-1)/20);
re_5=zeros(1,Len_y);
for i=0:1:(Len_y-1)/20
    re_1(1,i+1)=a(i*20+1,1);
    re_2(1,i+1)=a(i*20+1,2);
    re_3(1,i+1)=a(i*20+1,3);
    re_4(1,i+1)=a(i*20+1,4);
end
for i=1:1:Len_y
    re_5(1,i)=a(i,5);
end
    
h=0:20:500;
figure(1);
plot(h,re_1,'--','LineWidth',1);
axis([0 500 0 100]);
grid on
hold on
ylabel('Number of Nodes','FontSize',12)
xlabel('Time (s)','FontSize',12)
plot(h,re_2,'-*','LineWidth',1);
plot(h,re_3,'-^','LineWidth',1);
plot(h,re_4,'-o','LineWidth',1);
legend('re_1','re_2','re_3','re_4','re_5','re_6');
figure(2);
h2=0:1:500;
plot(h2,re_5,'-','LineWidth',1);
axis([0 500 0 1.2]);
grid on;
hold on;
ylabel('Detection Rate','FontSize',12)
xlabel('Time (s)','FontSize',12)