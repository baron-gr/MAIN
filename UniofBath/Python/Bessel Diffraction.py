import numpy as np
import matplotlib.pyplot as plt

def func(theta, x, m):
    return (np.cos(m*theta - x*(np.sin(theta))))
def x_val(r):
     return (np.pi/lamb)*(r/10)
def inten(r,j):
    return I_0*((2*j)/x)**2

#Trap
N=10000
a=0
b=np.pi
h=(b-a)/N

j_1 = []
r = np.linspace(-25,25,101)
theta=np.linspace(0,np.pi,N)

#Diffraction
lamb=0.5
I_0=1
Intensity=[]

for i in range(0,len(r-1)):
    total=0
    x=x_val(r[i])
    n=0
    if r[i] == 0:
        Intensity.append(1)
    else:
        for n in range (0, N-1):
            total = total + func((a+n*h), x, 1)
        Integral= h*(0.5*func(theta[1], b,1)+ total)
        Bessel= (1/np.pi)*Integral
        intensity_val= inten(x_val(r[i]), Bessel)
        Intensity.append(intensity_val)

plt.title('Diffraction intensity for Bessel m=1')
plt.plot(r,Intensity, 'r-')
plt.xlabel('r (m*10^-6)')
plt.ylabel('I(r)')
plt.show()