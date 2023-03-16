#Data
G=6.6741*10**-11
R_e=6.371*10**6
R_m=1.7371*10**6
M=5.9722*10**24
m=7.3420*10**22
R=3.8440*10**8
w=2.6617*10**-6

r1=10000
r2=40000
delta=10


def func(r):
   return (G*M/r**2)-(G*m/(R-r)**2)-(w**2*r)
    
def deriv(r1, r2):
   return (func(r2)-func(r1))/(r2-r1)

while(abs(delta)>0.0001):
    delta= -func(r2)/deriv(r1,r2)
    r3=r2+delta
    r1=r2
    r2=r3
r=r3/10**8
print('The lagrangian point from the centre of the Earth'' is {:1.4f}*10^8 m'.format(r))
