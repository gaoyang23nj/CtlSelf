#�޸�Ϊ����·��
#���ڲ�ͬ��alpha �������ſ��Ƶ����
library(bvpSolve)

output_data_name = "D:/24-CtlSelf/01-code/opt_solve_var_alpha.csv"


var_alpha <- seq(0.7, 1, by = 0.01)
optimal_t_on <- array(0, dim=length(var_alpha))
optimal_t_off <- array(0, dim=length(var_alpha))

for (var_alpha_index in 1:length(var_alpha))
{
#print(var_alpha_index)
print(var_alpha[var_alpha_index])

N <- 100
T <- 500

#betaҲ����Paper���\lambda
beta <- 0.004
rho <- 0.011

alpha <- var_alpha[var_alpha_index]
Um <- 1

time_granularity <- 0.1

# time
# x[1]=0,x[2]=1...,x[500]=499,x[501]=500
x <- seq(0, T, by = time_granularity)

# I[1],I[2],...,I[500],I[501] = 0
I <- array(0,dim=c(T+1))

final_I <- beta*N/(beta+rho)
I_t <- function(t) {
final_I - final_I * exp(-(beta+rho)*t)
}

#����I[1]=I_t(x[1])=I_t(0),I[2]=I_t(x[2])=I_t(1),...I[501]=I_t(x[501])=I_t(500)
i <- 1
while(i<=length(x)) {
I[i] = I_t(x[i])
i <- i+1
}

#��ӡ��I_t�ı仯
#plot(x, I, type = "l", main = "I, t", col = "red")

###########################
# BVP���⣺΢�ַ�����
# dot(M) = rho*I-beta*M-1/N*M*U
# dot(L) = L*(beta+1/N*U)-(1-alpha)
# U = 0 when alpha>=1/N*L*M  ; U=Um when alpha>=1/N*L*M
# M(t=0) = 0
# L(t+T) = 0

U_t <- function(t,M,L) {
# print(L)
# print(M)
# print(alpha)
tmp = 1/N*L*M
if(alpha > tmp){
return(0)
}else{
return(Um)
}
}

# M <- array(0,dim=c(T+1))
# L <- array(0,dim=c(T+1))

# y1=M dot(y1)=dot(M); 
# y2=L dot(y2)=dot(L); 
### partial(H)/partial(U) = 0 ; alpha=1/N*L*M; M* = alpha*N/L
# д�ɷ�������ʽ
fun <- function(x,y,pars) {
list(c(rho*I_t(x)-beta*y[1]-1/N*y[1]*U_t(x,y[1],y[2]),
y[2]*(beta+1/N*U_t(x,y[1],y[2]))-(1-alpha) )
)
}

# ���BVP����
sol1 <- bvpshoot(yini = c(0, NA), yend = c(NA, 0),
x = x, func = fun, guess = 0)

# BVP�������ֵ��
M <- sol1[,2]
L <- sol1[,3]

# ���� ���ſ��Ʊ���U^{*}(t) t=0,1,2...500 ��Ӧ��x[1],x[2]...x[501]
# ��ʼ��U[1],U[2],...,U[500],U[501] = 0
U <- array(0,dim=c(length(x)))
# ���� x:0,1,...500; idx:1,2,...501
i <- 1
while(i<=length(x)) {
tmp = 1/N*L[i]*M[i]
if(tmp>alpha){
U[i] = Um
}
i <- i+1
}


# д��data�ļ�
S <- N-M-I
all_solve <- cbind(x, M, I, S, L, U)
write.csv(all_solve, file = output_data_name, row.names = F)


###########################
# ��ӡ��ת�۵��λ��
i <- 1
oldvalue <- U[1]
points <- list()
while(i<=length(x)) {
if( U[i] != oldvalue ){
#��¼��Ӧ��idx
points[length(points)+1] <- i
oldvalue <- U[i]
}
i <- i+1
}

# x:0,1...500; idx:1,2,...501
turning_time <- unlist(points)-1
print('turning_time')
print(x[turning_time])
# ��0��1x
print(U[turning_time])
# ��1��0
print(U[turning_time+1])

##############################
if (length(x[turning_time])==2){
optimal_t_on[var_alpha_index] = x[turning_time][1]
optimal_t_off[var_alpha_index] = x[turning_time][2]
}
if ((length(x[turning_time])==1) && (U[turning_time][1]==0)){
optimal_t_on[var_alpha_index] = x[turning_time][1]
optimal_t_off[var_alpha_index] = T
}
if ((length(x[turning_time])==1) && (U[turning_time][1]==1)){
optimal_t_on[var_alpha_index] = 0
optimal_t_off[var_alpha_index] = x[turning_time][1]
}
if (length(x[turning_time])==0){
optimal_t_on[var_alpha_index] = 0
optimal_t_off[var_alpha_index] = 0
}


}
print(optimal_t_on)
print(optimal_t_off)
###############д����

all_solve_res <- cbind(var_alpha, optimal_t_on, optimal_t_off)
write.csv(all_solve_res, file = output_data_name, row.names = F)