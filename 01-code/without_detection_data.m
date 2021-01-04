clear all;
filename='E:\2020-2021第三学年\02论文\04-ICDCS\CtlSelf\01-code\without_detection_data.csv';
a=csvread(filename);
Len_x=size(a,2);
Len_y=size(a,1);
re_1=zeros(1,Len_y);
re_2=zeros(1,Len_y);
re_3=zeros(1,Len_y);
re_4=zeros(1,Len_y);
re_5=zeros(1,Len_y);
re_6=zeros(1,Len_y);
for i=1:1:Len_y
    re_1(1,i)=a(i,2);
    re_2(1,i)=a(i,3);
    re_3(1,i)=a(i,4);
    re_4(1,i)=a(i,5);
    re_5(1,i)=a(i,6);
    re_6(1,i)=a(i,7);
end
h=1:1:2500;
figure(1);
plot(h,re_1,'--','LineWidth',1);
axis([ 0 2550 0 120]);
grid on
hold on
ylabel('Number of Nodes','FontSize',12)
xlabel('Time (s)','FontSize',12)
plot(h,re_2,'--','LineWidth',1);
plot(h,re_3,'--','LineWidth',1);
plot(h,re_4,'-','LineWidth',1);
plot(h,re_5,'-','LineWidth',1);
plot(h,re_6,'-','LineWidth',1);
legend('re_1','re_2','re_3','re_4','re_5','re_6');