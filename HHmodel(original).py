import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy.signal as signal


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


def HH(points, t, para):
    [Cm, gNa, gK, gL, i_app] = para
    [V, n, m, h] = points
    return np.array([-(gNa * (m ** 3) * h * (V - VNa) + gK * (n ** 4) * (V - VK) + gL * (V - VL) - i_app) / Cm,
                     alphan(V) - (alphan(V) + betan(V)) * n, alpham(V) - (alpham(V) + betam(V)) * m,
                     alphah(V) - (alphah(V) + betah(V)) * h])

#units:V(mV), i_app(\muA/mm^2), C(\muF/mm^2), g(ms/mm^2)
V0 = -80#-65
VL = -49.4#-54.387
VK = -91.7#-77
VNa = 59.4#50
Cm=1
gNa=120
gK=36
gL=0.3
i_app=10
t = np.arange(0, 100, 0.01)
P1 = odeint(HH, (V0, alphan(V0) / (alphan(V0) + betan(V0)), alpham(V0) / (alpham(V0) + betam(V0)),
                 alphah(V0) / (alphah(V0) + betah(V0))), t, args=([Cm, gNa, gK, gL, i_app],))
print(P1[-1])
plt.plot(t,P1[:,0])
plt.show()

V1=np.array(P1[:,0])
N=P1[:,1]
M=P1[:,2]
H=P1[:,3]

Vmin=signal.argrelextrema(V1, np.less)
print(Vmin[0])
sumNa=0
sumK=0
sumL=0

j=0
for i in range(len(V1)):
    sumNa += gNa * (M[i] ** 3) * H[i] * (V1[i] - VNa)
    sumK+=gK * (N[i] ** 4) * (V1[i] - VK)
    sumL+=gL * (V1[i]- VL)
    if j==len(Vmin[0]):
        break
    if i==Vmin[0][j]:
        j+=1
        print(sumNa)
        print(sumK)
        print(sumL)
        sumNa=0
        sumK=0
        sumL=0