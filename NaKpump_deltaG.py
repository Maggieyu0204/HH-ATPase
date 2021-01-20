import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def NaKpump(points,t,para):
    [v, Nai,Ki,Nae,Ke,G,ADP,Pi,delta] = para #I(\muA/cm^2)
    [A, B, C, D] = points

    ATP = np.exp(G) * ADP * Pi
    #电压影响加在Na+上
    KdNae = KdNae0 * np.exp((1 + delta) * v / (3 * RTF))
    KdNai = KdNai0 * np.exp(delta * v / (3 * RTF))

    Naea = Nae / KdNae
    Naia = Nai / KdNai
    Kea = Ke / KdKe
    Kia = Ki / KdKi
    ATPa = ATP / KdATP

    alpha1p = k1p * Naia ** 3 / ((1 + Naia) ** 3 + (1 + Kia) ** 2 - 1)
    alpha2p = k2p
    alpha3p = k3p * Kea ** 2 / ((1 + Naea) ** 3 + (1 + Kea) ** 2 - 1)
    alpha4p = k4p * ATPa / (1 + ATPa)

    alpha1n = k1n * ADP
    alpha2n = k2n * Naea ** 3 / ((1 + Naea) ** 3 + (1 + Kea) ** 2 - 1)
    alpha3n = k3n * Pi*(10**(-4))/ (1 + ATPa)
    alpha4n = k4n * Kia ** 2 / ((1 + Naia) ** 3 + (1 + Kia) ** 2 - 1)

    dAdt = -alpha1p * A + alpha1n * B - alpha4n * A + alpha4p * D
    dBdt = -alpha2p * B + alpha2n * C - alpha1n * B + alpha1p * A
    dCdt = -alpha3p * C + alpha3n * D - alpha2n * C + alpha2p * B
    dDdt = -alpha4p * D + alpha4n * A - alpha3n * D + alpha3p * C

    ODE = np.array([dAdt, dBdt, dCdt, dDdt])
    return ODE

#constant
charge = 1.6 * 10 ** (-19)
NA = 6.02 * 10 ** (23)
RTF = 25.8


popt = np.loadtxt('parameter.txt')[0:14]
[k1p, k1n, k2p, k2n, k3p, k3n, k4p, k4n, KdNae0, KdNai0, KdKe, KdKi, KdATP, delta] = popt.tolist()
def multiply(i,k1p, k1n, k2p, k2n, k3p, k3n, k4p, k4n):
    k1p=i*k1p
    k1n=i*k1n
    k2p=i*k2p
    k2n=i*k2n
    k3p=i*k3p
    k3n=i*k3n
    k4p=i*k4p
    k4n=i*k4n

multiply(200,k1p, k1n, k2p, k2n, k3p, k3n, k4p, k4n)


dt=0.01

J=[]
for G in np.arange(0,10,1):
    J1=[]
    for v in np.arange(-120,60,5):
        # [v, Nai,Ki,Nae,Ke,ATP,ADP,Pi,delta]
        para = [v, 50, 140, 150,5.4,G, 0.05, 4.2, delta]
        time1 = np.arange(0, 200, dt)
        results1 = odeint(NaKpump, [1,0,0,0], time1, args=(para,))
        B=results1[-1][1]
        C=results1[-1][2]
        KdNae = KdNae0 * np.exp((1 + delta) * v / (3 * RTF))
        Naea = 150 / KdNae
        Kea = 5.4 / KdKe
        alpha2p = k2p
        alpha2n = k2n * Naea ** 3 / ((1 + Naea) ** 3 + (1 + Kea) ** 2 - 1)
        Jp = (alpha2p * B - alpha2n * C) * 1000
        print(v,alpha2n,alpha2p,B,C)
        J1.append(Jp)
    J.append(J1)

x=np.arange(-120,60,5)
plt.plot(x,J[0],label='0')
plt.plot(x,J[1],label='1')
plt.plot(x,J[2],label='2')
plt.plot(x,J[3],label='3')
plt.plot(x,J[4],label='4')
plt.plot(x,J[5],label='5')
plt.plot(x,J[6],label='6')
plt.legend()
plt.show()