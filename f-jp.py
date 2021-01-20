import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy.signal as signal
from mpl_toolkits.mplot3d import Axes3D


def HHmodel2(points, t, para):
    [I, k1p, k1n, k2p, k2n, k3p, k3n, k4p, k4n] = para  # I(\muA/cm^2)
    [v, Nai, Ki, n, m, h, A, B, C, D] = points

    EK = RTF * np.log(Ke / Ki)  # mV
    ENa = RTF * np.log(Nae / Nai)  # mV

    alphan = 0.01 * (v + 50) / (1 - np.exp(-(v + 50) / 10))
    betan = 0.125 * np.exp(-(v + 60) / 80)
    alpham = 0.1 * (v + 35) / (1 - np.exp(-(v + 35) / 10))
    betam = 4 * np.exp(-(v + 60) / 18)
    alphah = 0.07 * np.exp(-(v + 60) / 20)
    betah = 1 / (np.exp(-(v + 30) / 10) + 1)

    Nae1a = Nae / (KdNae0 * np.exp((1 + delta) * v / RTF))
    Nae2a = Nae / KdNae
    Nai1a = Nai / (KdNai0 * np.exp(delta * v / RTF))
    Nai2a = Nai / KdNai
    Kea = Ke / KdKe
    Kia = Ki / KdKi
    ATPa = ATP / KdATP

    alpha1p = k1p * Nai1a * (Nai2a ** 2) / (Nai1a * (Nai2a ** 2) + (1 + Nai2a) ** 2 + (1 + Kia) ** 2 - 1)
    alpha2p = k2p
    alpha3p = k3p * Kea ** 2 / (Nae1a * (Nae2a ** 2) + (1 + Nae2a) ** 2 + (1 + Kea) ** 2 - 1)
    alpha4p = k4p * ATPa / (1 + ATPa)

    alpha1n = k1n * ADP
    alpha2n = k2n * Nae1a * (Nae2a ** 2) / (Nae1a * (Nae2a ** 2) + (1 + Nae2a) ** 2 + (1 + Kea) ** 2 - 1)
    alpha3n = k3n * Pi * (10 ** (-4)) / (1 + ATPa)
    alpha4n = k4n * Kia ** 2 / (Nai1a * (Nai2a ** 2) + (1 + Nai2a) ** 2 + (1 + Kia) ** 2 - 1)

    Jp = alpha2p * B - alpha2n * C

    dvdt = (gl * (El - v) + gNa * m ** 3 * h * (ENa - v) + gK * n ** 4 * (
            EK - v) - sigma_pump * charge * Jp * (10 ** 17) + I) / Cm
    dNaidt = -10 * sigma_pump * Jp * (10 ** 22) / (R * NA) - 2 * gNa * (m ** 3) * h * (v - ENa) * (10 ** 5) / (
            charge * NA * R)
    dKidt = 10 * sigma_pump * Jp * (10 ** 22) / (R * NA) - 2 * gK * (n ** 4) * (v - EK) * (10 ** 5) / (charge * NA * R)
    dndt = alphan * (1 - n) - betan * n
    dmdt = alpham * (1 - m) - betam * m
    dhdt = alphah * (1 - h) - betah * h
    dAdt = -alpha1p * A + alpha1n * B - alpha4n * A + alpha4p * D
    dBdt = -alpha2p * B + alpha2n * C - alpha1n * B + alpha1p * A
    dCdt = -alpha3p * C + alpha3n * D - alpha2n * C + alpha2p * B
    dDdt = -alpha4p * D + alpha4n * A - alpha3n * D + alpha3p * C

    ODE = np.array([dvdt, dNaidt, dKidt, dndt, dmdt, dhdt, dAdt, dBdt, dCdt, dDdt])
    return ODE


G = 12 #G/RT

# concentration(mMol/L)
Nae = 140  # 437
Ke = 5  # 8.46  # 24
ADP = 0.05
Pi = 0.8  # 4.2
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
[k1p, k1n, k2p, k2n, k3p, k3n, k4p, k4n, KdNai0, KdNae0, KdNai, KdNae, KdKi, KdKe, KdATP, delta,
 sigma_pump] = popt.tolist()


def find_increase(I, turn, k1p, k1n, k2p, k2n, k3p, k3n, k4p, k4n):
    k1p = turn * k1p
    k1n = turn * k1n
    k2p = turn * k2p
    k2n = turn * k2n
    k3p = turn * k3p
    k3n = turn * k3n
    k4p = turn * k4p
    k4n = turn * k4n
    # [v, Nai, Ki, n, m, h, A, B, C, D]
    initial = [-60, 15, 140, 0.5, 0, 1, 1, 0, 0, 0]
    dt = 0.01
    time = np.arange(0, 300, dt)
    results = odeint(HHmodel2, initial, time, args=([I, k1p, k1n, k2p, k2n, k3p, k3n, k4p, k4n],))
    V1 = np.array(results[:, 0])[10000:]
    NA = np.array(results[:, 1])[10000:]
    Vmax = signal.argrelextrema(V1, np.greater)
    Vmin = signal.argrelextrema(V1, np.less)
    Namax = signal.argrelextrema(NA, np.greater)
    if len(Vmax[0]) < 4 or len(Vmin[0]) < 4 or V1[Vmax[0][-1]] - V1[Vmin[0][-1]] < 30 or Vmin[0][-1] < 17000:
        return None
    else:
        if abs(NA[Namax[0][-1]] - NA[Namax[0][0]]) < 0.1:
            f = 100000 / (Vmin[0][-1] - Vmin[0][-2])
            J = 0
            B = np.array(results[:, 7])[10000:]
            C = np.array(results[:, 8])[10000:]
            for i in range(len(V1)):
                Nae1a = Nae / (KdNae0 * np.exp((1 + delta) * V1[i] / (RTF)))
                Nae2a = Nae / KdNae
                Kea = Ke / KdKe
                alpha2p = k2p
                alpha2n = k2n * Nae1a * (Nae2a ** 2) / (Nae1a * (Nae2a ** 2) + (1 + Nae2a) ** 2 + (1 + Kea) ** 2 - 1)
                Jp = (alpha2p * B[i] - alpha2n * C[i]) / 20
                J += Jp
            return [f, J]
        else:
            if NA[-1] - NA[0] > 0:
                return [0, 1]
            else:
                return [0, -1]


I = np.arange(30, 200, 1)
f = []
J = []
for i in I:
    find = False
    for turn in np.arange(1, 61, 0.5):
        a = find_increase(i, turn, k1p, k1n, k2p, k2n, k3p, k3n, k4p, k4n)
        if a != None:
            if a[0] != 0:
                f.append(a[0])
                J.append(a[1])
                print(i, turn, a)
                find = True
                break
    if find == False:
        f.append(0)
        J.append(0)
        print(i, [0, 0])
