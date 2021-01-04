all = csvread('result_20200922154512_time_seq_uniform_detect_M_N_U.tmp');
s = size(all);
%% t0:0,10,20,...490, // t1:1,11,21,...491
cnt = 0;
res = zeros(50,50);
for i = 1:s(1,1)-1
    t0 = all(i,1);
    t1 = all(i,2);
    idx0 = floor(t0/10)+1;
    idx1 = floor(t1/10)+1;
    if res(idx0,idx1)==0
        res(idx0,idx1)=all(i,5);
        cnt = cnt + 1;
    else
        fprintf('Duplicate Err!!!\n')
    end
end
cnt

%% t0:0,10,20,...490, // repmat( A , m , n )
t0 = 0:10:490;
t0 = t0';
t0 = repmat(t0,1,50);

%% t1:1,11,21,...491， // 
t1 = 1:10:491;
t1 = repmat(t1,50,1);

%% 存在一半的零值
for i=1:50
    for j=1:50
        if i>j
            res(i,j)= res(j,i);
        end
    end
end

%% surf
%meshc(t0,t1,res);
surfc(t0,t1,res);
% all(end,:)
hold on;
%plot3(floor(all(end,1)/10) + 1, floor(all(end,2)/10) + 1, all(end,3),'k*','MarkerSize',5)
plot3(all(end,1), all(end,2), all(end,5),'k*','MarkerSize',10);
plot3(all(end,1), all(end,2), min(min(res)),'ko','MarkerSize',5);
xlabel('t0');
ylabel('t1');
zlabel('cost');
set(gca, 'Fontname', 'Times New Roman','FontSize',15);