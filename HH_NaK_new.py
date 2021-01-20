import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def HHmodel(points, t, para):
    [I] = para  # I(\muA/cm^2)
    [v, Nai, Ki, n, m, h, A, B, C, D] = points

    EK = RTF * np.log(Ke / Ki)  # mV
    ENa = RTF * np.log(Nae / Nai)  # mV

    alphan = 0.01 * (v + 50) / (1 - np.exp(-(v + 50) / 10))
    betan = 0.125 * np.exp(-(v + 60) / 80)
    alpham = 0.1 * (v + 35) / (1 - np.exp(-(v + 35) / 10))
    betam = 4 * np.exp(-(v + 60) / 18)
    alphah = 0.07 * np.exp(-(v + 60) / 20)
    betah = 1 / (np.exp(-(v + 30) / 10) + 1)

    Nae1a = Nae / (KdNae0*np.exp((1 + delta) * v / RTF))
    Nae2a = Nae / KdNae
    Nai1a = Nai / (KdNai0*np.exp(delta * v / RTF))
    Nai2a = Nai / KdNai
    Kea = Ke / KdKe
    Kia = Ki / KdKi
    ATPa = ATP / KdATP

    alpha1p = k1p * Nai1a *(Nai2a** 2) / (Nai1a *(Nai2a** 2)+(1 + Nai2a) ** 2 + (1 + Kia) ** 2 - 1)
    alpha2p = k2p
    alpha3p = k3p * Kea ** 2 / (Nae1a *(Nae2a** 2)+(1 + Nae2a) ** 2 + (1 + Kea) ** 2 - 1)
    alpha4p = k4p * ATPa / (1 + ATPa)

    alpha1n = k1n * ADP
    alpha2n = k2n * Nae1a *(Nae2a** 2) /  (Nae1a *(Nae2a** 2)+(1 + Nae2a) ** 2 + (1 + Kea) ** 2 - 1)
    alpha3n = k3n * Pi * (10 ** (-4)) / (1 + ATPa)
    alpha4n = k4n * Kia ** 2 / (Nai1a *(Nai2a** 2)+(1 + Nai2a) ** 2 + (1 + Kia) ** 2 - 1)

    Jp =alpha2p * B - alpha2n * C

    dvdt = (gl * (El - v) + gNa * m ** 3 * h * (ENa - v) + gK * n ** 4 * (
            EK - v) - sigma_pump * charge * Jp* (10 ** 17) + I) / Cm
    dNaidt = -6 * sigma_pump * Jp * (10 ** 22) / (R * NA) - 2 * gNa * (m ** 3) * h * (v - ENa) * (10 ** 5) / (
           charge * NA * R)
    dKidt = 4 * sigma_pump * Jp * (10 ** 22) / (R * NA) - 2 * gK * (n ** 4) * (v - EK) * (10 ** 5) / (charge * NA * R)
    dndt = alphan * (1 - n) - betan * n
    dmdt = alpham * (1 - m) - betam * m
    dhdt = alphah * (1 - h) - betah * h
    dAdt = -alpha1p * A + alpha1n * B - alpha4n * A + alpha4p * D
    dBdt = -alpha2p * B + alpha2n * C - alpha1n * B + alpha1p * A
    dCdt = -alpha3p * C + alpha3n * D - alpha2n * C + alpha2p * B
    dDdt = -alpha4p * D + alpha4n * A - alpha3n * D + alpha3p * C

    ODE = np.array([dvdt, dNaidt, dKidt, dndt, dmdt, dhdt, dAdt, dBdt, dCdt, dDdt])
    return ODE

def alphan(V):
    return 0.01 * (V + 50) / (1 - np.exp(-0.1 * (V + 50)))


def betan(V):
    return 0.125 * np.exp(-0.0125 * (V + 60))


def alpham(V):
    return 0.1 * (V + 35) / (1 - np.exp(-0.1 * (V + 35)))


def betam(V):
    return 4 * np.exp(-0.0556 * (V + 60))


def alphah(V):
    return 0.07 * np.exp(-0.05 * (V + 60))


def betah(V):
    return 1 / (1 + np.exp(-0.1 * (V + 30)))

G = 6

# concentration(mMol/L)
Nae = 150  # 437
Ke = 4 #8.46  # 24
ADP = 0.05
Pi = 0.8#4.2
ATP = np.exp(G) * ADP * Pi

# constant
charge = 1.6 * 10 ** (-19)
NA = 6.02 * 10 ** (23)
RTF = 25.8

# 轴突半径(A)
R = 10000
# V(mV)
El = -49.4  # -49.4
# g(ms/cm^2)=
gl = 0.3
gNa = 120
gK = 36
# C(muF/cm^2)
Cm = 1

popt = np.loadtxt('parameter_new.txt')[0:17]
[k1p, k1n, k2p, k2n, k3p, k3n, k4p, k4n, KdNai0, KdNae0, KdNai, KdNae, KdKi, KdKe, KdATP, delta,sigma_pump] = popt.tolist()

def multiply(i,k1p, k1n, k2p, k2n, k3p, k3n, k4p, k4n):
    k1p=i*k1p
    k1n=i*k1n
    k2p=i*k2p
    k2n=i*k2n
    k3p=i*k3p
    k3n=i*k3n
    k4p=i*k4p
    k4n=i*k4n
    return [k1p, k1n, k2p, k2n, k3p, k3n, k4p, k4n]

[k1p, k1n, k2p, k2n, k3p, k3n, k4p, k4n]=multiply(1,k1p, k1n, k2p, k2n, k3p, k3n, k4p, k4n)


# [v, Nai, Ki, n, m, h, A, B, C, D]
V0=-80
n=alphan(V0) / (alphan(V0) + betan(V0))
m=alpham(V0) / (alpham(V0) + betam(V0))
h=alphah(V0) / (alphah(V0) + betah(V0))
initial2 = [V0, 15, 140, n, m,h, 1, 0, 0, 0]
dt = 0.01
t=0.2
I=40
time1 = np.arange(0, 500*t, dt)
time2=np.arange(500*t, 500,dt)
time3 = np.arange(500 , 500+500*t, dt)
time4 = np.arange(500+500*t, 1000, dt)
time5 = np.arange(1000, 1000+500*t, dt)
time6 = np.arange(1000+500 * t, 1500, dt)
time7 = np.arange(1500, 1500+500*t, dt)
time8 = np.arange(1500+500 * t, 2000, dt)
result1 = odeint(HHmodel, initial2, time1, args=([I],))
result2 = odeint(HHmodel, result1[-1], time2, args=([0],))
result3 = odeint(HHmodel, result2[-1], time1, args=([I],))
result4 = odeint(HHmodel, result3[-1], time2, args=([0],))
result5 = odeint(HHmodel, result4[-1], time1, args=([I],))
result6 = odeint(HHmodel, result5[-1], time2, args=([0],))
result7 = odeint(HHmodel, result6[-1], time1, args=([I],))
result8 = odeint(HHmodel, result7[-1], time2, args=([0],))
time = np.hstack((time1, time2, time3,time4,time5,time6,time7,time8))  # ms
results = np.vstack((result1, result2, result3,result4,result5,result6,result7,result8))
V= np.array(results[:, 0])

#time=time2
#result=results2
J = []
for i in range(len(time)):
    v = results[i, 0]
    B = results[i, 7]
    C = results[i, 8]
    Nae1a = Nae / (KdNae0 * np.exp((1 + delta) * v / (RTF)))
    Nae2a = Nae / KdNae
    Kea = Ke / KdKe
    alpha2p = k2p
    alpha2n = k2n * Nae1a *(Nae2a** 2) /  (Nae1a *(Nae2a** 2)+(1 + Nae2a) ** 2 + (1 + Kea) ** 2 - 1)
    Jp = (alpha2p * B - alpha2n * C) *1000
    if i % 100 == 0:
        print(i, v, Jp)
    J.append(Jp)

V1=results[:,0]
NA=results[:,1]
K=results[:,2]
N=results[:,3]
M=results[:,4]
H=results[:,5]

fig = plt.figure()
ax1 = fig.add_subplot(411)
ax1.plot(time, results[:, 0], label='voltage')
#ax1.plot(time, J, label='dynamic')
# ax1.plot(time,result[:,3],label='n')
# ax1.plot(time,result[:,4],label='m')
# ax1.plot(time,result[:,5],label='h')
#ax1.plot(time, result[:, 0], label='voltage')
#ax1.set_ylim((-100, 50))
plt.legend()
ax2 = fig.add_subplot(412)
ax2.plot(time, NA, label='Na')
#ax2.plot(time, result[:, 7], label='B')
#ax2.plot(time, result[:, 8], label='C')
#ax2.plot(time, result[:, 9], label='D')
plt.legend()
ax3 = fig.add_subplot(413)
ax3.plot(time, K, label='K')
plt.legend()
ax4 = fig.add_subplot(414)
ax4.plot(time, J, label='dynamic')
plt.legend()
plt.show()
# [I, delta] = para
# [v, Nai, Ki, n, m, h, A, B, C, D] = points
