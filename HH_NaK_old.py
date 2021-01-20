import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy.signal as signal


def HHmodel1(points, t, para):
    [I] = para  # I(\muA/cm^2)
    [v, n, m, h] = points
    EK = -72.4
    ENa = 55.9
    alphan = 0.01 * (v + 50) / (1 - np.exp(-(v + 50) / 10))
    betan = 0.125 * np.exp(-(v + 60) / 80)
    alpham = 0.1 * (v + 35) / (1 - np.exp(-(v + 35) / 10))
    betam = 4 * np.exp(-(v + 60) / 18)
    alphah = 0.07 * np.exp(-(v + 60) / 20)
    betah = 1 / (np.exp(-(v + 30) / 10) + 1)
    dvdt = (gl * (El - v) + gNa * m ** 3 * h * (ENa - v) + gK * n ** 4 * (
            EK - v) + I) / Cm
    dndt = alphan * (1 - n) - betan * n
    dmdt = alpham * (1 - m) - betam * m
    dhdt = alphah * (1 - h) - betah * h
    ODE = np.array([dvdt, dndt, dmdt, dhdt])
    return ODE


def HHmodel2(points, t, para):
    [I, delta] = para  # I(\muA/cm^2)
    [v, Nai, Ki, n, m, h, A, B, C, D] = points

    EK = RTF * np.log(Ke / Ki)  # mV
    ENa = RTF * np.log(Nae / Nai)  # mV

    alphan = 0.01 * (v + 50) / (1 - np.exp(-(v + 50) / 10))
    betan = 0.125 * np.exp(-(v + 60) / 80)
    alpham = 0.1 * (v + 35) / (1 - np.exp(-(v + 35) / 10))
    betam = 4 * np.exp(-(v + 60) / 18)
    alphah = 0.07 * np.exp(-(v + 60) / 20)
    betah = 1 / (np.exp(-(v + 30) / 10) + 1)

    # 电压影响加在Na+上
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
    alpha3n = k3n * Pi * (10 ** (-4)) / (1 + ATPa)
    alpha4n = k4n * Kia ** 2 / ((1 + Naia) ** 3 + (1 + Kia) ** 2 - 1)

    Jp = alpha2p * B - alpha2n * C

    dvdt = (gl * (El - v) + gNa * m ** 3 * h * (ENa - v) + gK * n ** 4 * (
            EK - v) - sigma_pump * charge * Jp * (10 ** 17) + I) / Cm
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


G = 8

# concentration(mMol/L)
Nae = 150  # 437
Ke = 5  # 24
ADP = 0.05
Pi = 0.8
ATP = np.exp(G) * ADP * Pi

# constant
charge = 1.6 * 10 ** (-19)
NA = 6.02 * 10 ** (23)
RTF = 25.8

# 轴突半径(A)
R = 10000
# NaKpump数量(\mum^-2)
sigma_pump = 2000
# V(mV)
El = -49.4  # -49.4
# g(ms/cm^2)=
gl = 0.3
gNa = 120
gK = 36
# C(muF/cm^2)
Cm = 1

popt = np.loadtxt('parameter.txt')[0:14]
[k1p, k1n, k2p, k2n, k3p, k3n, k4p, k4n, KdNae0, KdNai0, KdKe, KdKi, KdATP, delta] = popt.tolist()


def multiply(i, k1p, k1n, k2p, k2n, k3p, k3n, k4p, k4n):
    k1p = i * k1p
    k1n = i * k1n
    k2p = i * k2p
    k2n = i * k2n
    k3p = i * k3p
    k3n = i * k3n
    k4p = i * k4p
    k4n = i * k4n
    return [k1p, k1n, k2p, k2n, k3p, k3n, k4p, k4n]


[k1p, k1n, k2p, k2n, k3p, k3n, k4p, k4n] = multiply(30, k1p, k1n, k2p, k2n, k3p, k3n, k4p, k4n)

initial1 = [-70, 0.5, 0, 1]
# [v, Nai, Ki, n, m, h, A, B, C, D]
initial2 = [-70, 15, 140, 0.5, 0, 1, 1, 0, 0, 0]

dt = 0.01

time1 = np.arange(0, 50, dt)
I = 25
results11 = odeint(HHmodel1, initial1, time1, args=([I],))
results12 = odeint(HHmodel2, initial2, time1, args=([I, delta],))

time2 = np.arange(50, 1000, dt)
I = 0
results21 = odeint(HHmodel1, results11[-1], time2, args=([I],))
results22 = odeint(HHmodel2, results12[-1], time2, args=([I, delta],))

time3 = np.arange(1000, 1050, dt)
I = 25
results32 = odeint(HHmodel2, results22[-1], time3, args=([I, delta],))
print(results3[-1])

time = np.hstack((time1, time2, time3))  # ms
result1 = np.vstack((results11, results21))
result2 = np.vstack((results12, results22, results32))
# time=time2
# result=results2

J = []
for i in range(len(time)):
    v = result2[i, 0]
    n = result2[i, 3]
    m = result2[i, 4]
    h = result2[i, 5]
    KdNae = KdNae0 * np.exp((1 + delta) * v / (3 * RTF))
    B = result2[i, 7]
    C = result2[i, 8]
    Naea = Nae / KdNae
    Kea = Ke / KdKe
    alpha2p = k2p
    alpha2n = k2n * Naea ** 3 / ((1 + Naea) ** 3 + (1 + Kea) ** 2 - 1)
    Jp = (alpha2p * B - alpha2n * C) * (1000)
    if i % 100 == 0:
        print(i, v, n, m, h, Jp)
    J.append(Jp)

V1 = result2[:, 0]
NA = result2[:, 1]
K = result2[:, 2]
N = result2[:, 3]
M = result2[:, 4]
H = result2[:, 5]

Vmin = signal.argrelextrema(V1, np.less)
print(Vmin[0])
sumNa = 0
sumK = 0
sumL = 0

j = 0
for i in range(len(V1)):
    EK = RTF * np.log(Ke / K[i])  # mV
    ENa = RTF * np.log(Nae / NA[i])  # mV
    sumNa += gNa * (M[i] ** 3) * H[i] * (V1[i] - ENa)
    sumK += gK * (N[i] ** 4) * (V1[i] - EK)
    sumL += gl * (V1[i] - El)
    if j == len(Vmin[0]):
        break
    if i == Vmin[0][j]:
        j += 1
        print(sumNa)
        print(sumK)
        print(sumL)
        sumNa = 0
        sumK = 0
        sumL = 0

fig = plt.figure()
ax1 = fig.add_subplot(411)
# ax1.plot(time, result1[:, 0], label='voltage1')
ax1.plot(time, result2[:, 0], label='voltage')
# ax1.plot(time,result[:,3],label='n')
# ax1.plot(time,result[:,4],label='m')
# ax1.plot(time,result[:,5],label='h')
# ax1.plot(time, result[:, 0], label='voltage')
# ax1.set_ylim((-100, 50))
plt.legend()
ax2 = fig.add_subplot(412)
ax2.plot(time, result2[:, 1], label='Na')
# ax2.plot(time, result[:, 7], label='B')
# ax2.plot(time, result[:, 8], label='C')
# ax2.plot(time, result[:, 9], label='D')
plt.legend()
ax3 = fig.add_subplot(413)
ax3.plot(time, result2[:, 2], label='K')
plt.legend()
ax4 = fig.add_subplot(414)
plt.plot(time, J, label='dynamic')
plt.legend()
plt.show()
# [I, delta] = para
# [v, Nai, Ki, n, m, h, A, B, C, D] = points
