import numpy as np
import matplotlib.pyplot as plt

def func(theta, x, m):
    return (1/np.pi)*(np.cos(m*theta - x*(np.sin(theta))))

j_0 = []
j_1 = []
j_2 = []

#Trap
N=10000
a=0
b=np.pi
h=(b-a)/N

#Bessel
A=0
B=20
#Smoothness coeff
S=100
H=(B-A)/S
X=A+H*np.arange(0,S+1)


while A<=B:
    total=0
    n=1
    while n<N:
        total = total + func((a+n*h), A, 0)
        n=n+1
    j_0.append(0.5*h*(func(a, A, 0)+func(b, A, 0)+2*(total)))
    A=A+H

A=0
while A<=B:
    total=0
    n=1
    while n<N:
        total = total + func((a+n*h), A, 1)
        n=n+1
    j_1.append(0.5*h*(func(a, A, 1)+func(b, A, 1)+2*(total)))
    A=A+H

A=0
while A<=B:
    total=0
    n=1
    while n<N:
        total = total + func((a+n*h), A, 2)
        n=n+1
    j_2.append(0.5*h*(func(a, A, 2)+func(b, A, 2)+2*(total)))
    A=A+H

plt.title('Bessel function for m=0,1,2')
plt.plot(X,j_0, 'r-', label='m=0')
plt.plot(X,j_1, 'b-', label='m=1')
plt.plot(X,j_2, 'c-', label='m=2')
plt.xlabel('X')
plt.ylabel('J_m(X)')
plt.legend(loc='upper right')
plt.show()