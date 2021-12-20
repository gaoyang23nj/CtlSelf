# 利用Euler+线性插值方法求解 MASS2021的最优自私检测问题
import scipy.integrate
import numpy as np

T = 500
lam = 0.004
N = 100
rho = 0.011
alpha = 0.9
Um = 1

# D' = -(lam+U(t)/N)D+rho*I(t)
# lambda_D' = lambda_D*(lam+U(t)/N)-(1-alpha)
# D(0)=0
# lambda_D(T) = 0
# U = 1-sgn(alpha-lambda_D(t)*D(t)/N)
# I(t) = (lam*N/(lam+rho) )* (1-exp(-(lam+rho)*t))

def I(t):
    res = (lam*N/(lam+rho)) * (1-np.math.exp(-(lam+rho)*t))
    return res

def U(D,lamD):
    res = Um
    if alpha >= D * lamD / N:
        res = 0
    return res

x = np.linspace(0, T, 500)
y = np.zeros((2, x.size))
target_y = np.zeros((2, x.size))

# Assume lambda_D[0] = 0
# lambda_D' = lambda_D*(lam+0/N)-(1-alpha)=lambda_D*lam-(1-alpha)
# lambda_D(t) = C*exp(lam*x)+(1-alpha)/lam
# 0 = C*exp(lam*T)+(1-alpha)/lam
C = -(1-alpha)/lam /np.exp(lam*T)
print(C)
init_lambdaD = C*np.math.exp(lam*0)+(1-alpha)/lam
print(init_lambdaD)

y[0,0] = 0
y[1,0] = init_lambdaD
for i in range(x.size):
    # 计算瞬时梯度
    derivateD = -(lam+U(y[0,i],y[1,i])/N)*y[0,i]+rho*I(x[i])
    derivatelamD = (lam+U(y[0,i],y[1,i])/N)*y[1,i]-(1-alpha)
    if i<x.size-1:
        y[0,i+1] = y[0,i]+derivateD*(x[i+1]-x[i])
        y[1,i+1] = y[1,i]+derivatelamD*(x[i+1]-x[i])
print(y)

ll = []
ll.append((y[1,0], y[1,x.size-1]))
print(ll[-1])

y[0,0] = 0
init_lambdaD = 13.11

k=100
while k>0:
    k = k-1
    # Euler method
    # 计算根据现在的init得到怎样的终端值
    y[1,0] = init_lambdaD
    for i in range(x.size):
        # 计算瞬时梯度
        derivateD = -(lam+U(y[0,i],y[1,i])/N)*y[0,i]+rho*I(x[i])
        derivatelamD = (lam+U(y[0,i],y[1,i])/N)*y[1,i]-(1-alpha)
        if i<x.size-1:
            y[0,i+1] = y[0,i]+derivateD*(x[i+1]-x[i])
            y[1,i+1] = y[1,i]+derivatelamD*(x[i+1]-x[i])
        # y[1,x.size-1]
    ll.append((y[1,0], y[1,x.size-1]))
    print(ll[-1])
    if np.math.fabs(ll[-1][1]-ll[-2][1]) < 0.00001:
        print('condition ok!')
        break
    # 找到下一个init_lambdaD
    init_lambdaD = (ll[-1][0]-ll[-2][0])/(ll[-1][1]-ll[-2][1])*(target_y[1,x.size-1]-ll[-2][1])+ll[-2][0]

# print(y)
print(k)


x_plot = np.linspace(0, T, 100)
# y_plot_a = res_a.sol(x_plot)[0]

import matplotlib.pyplot as plt
plt.plot(x, y[0], label='y_0')
plt.plot(x, y[1], label='y_1')
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()
