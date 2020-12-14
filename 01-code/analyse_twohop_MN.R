library(bvpSolve)

N <- 100
T <- 500

beta <- 0.004
rho <- 0.01

alpha <- 0.2
Um <- 1

# 0,1,...,500
x <- seq(0, T, by = 1)

# I(0),I(1),...,I(500)
I <- array(0,dim=c(T+1))

final_I <- beta*N/(beta+rho)

i <- 1
while(i<=length(x)) {
I[i] = I_t(i-1)
i <- i+1
}

I_t <- function(t) {
final_I - final_I * exp(-(beta+rho)*t)
}

#plot(x, I, type = "l", main = "I, t", col = "red")

###########################

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
# print(tmp)
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
fun<- function(x,y,pars) {
list(c(rho*I_t(x)-beta*y[1]-1/N*y[1]*U_t(x,y[1],y[2]),
y[2]*(beta+1/N*U_t(x,y[1],y[2]))-(1-alpha) )
)
}

sol1 <- bvpshoot(yini = c(0, NA), yend = c(NA, 0),
x = x, func = fun, guess = 0)

M <- sol1[,2]
L <- sol1[,3]
U <- array(0,dim=c(T+1))

# 控制 x:0,1,...500; idx:1,2,...501
i <- 1
while(i<=length(x)) {
tmp = 1/N*L[i]*M[i]
if(tmp>alpha){
U[i] = Um
}
i <- i+1
}

i <- 1
oldvalue <- U[1]
points <- list()
while(i<=length(x)) {
if( U[i] != oldvalue ){
#记录对应的idx
points[length(points)+1] <- i
oldvalue <- U[i]
}
i <- i+1
}

# x:0,1...500; idx:1,2,...501
turning_time <- unlist(points)-1
print('turning_time')
print(turning_time)
print(U[turning_time])
print(U[turning_time+1])

S <- N-M-I
all_solve <- cbind(sol1, U, I, S)
write.csv(all_solve, file = "D:/24-ODEpaper/03-OSESim_twohop_BVP_MN_better/opt_solve.csv",row.names = F)




##############################
plot(x, M, type = "l", xlab="time", ylab="value", lwd = 2, col = "red", 
main= "state")

lines(x, L, type = "l", lwd = 2, col = "green")

title("state")

legend(20,50,c("M(t)", "L(t)", "1/N*L*M"), lwd=c(2,2,2), col=c("red","green","black"), y.intersp=1.5)

##############################
criteria <- 1/N*M*L
plot(x, criteria, type = "l", lwd = 2, col = "black")

line_alpha <- rep(alpha, length(x))
lines(x, line_alpha, type = "l", lwd = 2, col = "yellow")

##############################
plot(x, U, type = "l", xlab="time", ylab="control", lwd=1, col = "blue", 
main = "control")

title("control")








