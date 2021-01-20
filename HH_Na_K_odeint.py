import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def HHmodel(points,t,para):
    # Jp=4*NaKpump(Nai,Ki,v)*Hz: Hz (constant over dt) #NaKpump flux

    [v,Nai,Ki,n,m,h,A,B,C,D]=points
    [I,delta]=para

    EK = RTF * np.log(Ke / Ki) #mV
    ENa = RTF * np.log(Nae / Nai) #mV
    # A=Ei(ATP), B=EiP(3Na), C=EeP(2K), D=Ee(ATP)


    alphan = 0.01 * (v + 50) / (1 - np.exp(-(v + 50) / 10))
    betan = 0.125 * np.exp(-(v + 60) / 80)
    alpham = 0.1 * (v + 35) / (1 - np.exp(-(v + 35) / 10))
    betam = 4 * np.exp(-(v + 60) / 18)
    alphah = 0.07 * np.exp(-(v + 60) / 20)
    betah = 1 / (np.exp(-(v + 30) / 10) + 1)

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
    alpha3n = k3n * Pi / (1 + ATPa)
    alpha4n = k4n * Kia ** 2 / ((1 + Naia) ** 3 + (1 + Kia) ** 2 - 1)

    Jp = alpha2p * B - alpha2n*C

    dvdt = (gl * (El - v) + gNa * m ** 3 * h * (ENa - v) + gK * n ** 4 * (
                EK - v) - sigma_pump * charge * Jp + I) / Cm
    dNaidt = -6 * sigma_pump / (R * NA) * Jp - 2 * gNa * m ** 3 * h * (v - ENa) / (
                charge * NA * R)
    dKidt = 4 * sigma_pump / (R * NA) * Jp - 2 * gK * n ** 4 * (v - EK) / (charge * NA * R)
    dmdt = alpham * (1 - m) - betam * m
    dndt = alphan * (1 - n) - betan * n
    dhdt = alphah * (1 - h) - betah * h
    dAdt = -alpha1p * A + alpha1n * B - alpha4n * A + alpha4p * D
    dBdt = -alpha2p * B + alpha2n * C - alpha1n * B + alpha1p * A
    dCdt = -alpha3p * C + alpha3n * D - alpha2n * C + alpha2p * B
    dDdt = -alpha4p * D + alpha4n * A - alpha3n * D + alpha3p * C

    ODE=np.array([dvdt,dNaidt,dKidt,dmdt,dndt,dhdt,dAdt,dBdt,dCdt,dDdt])
    return ODE

initial=[-60,50,397,0.5,0,1,1,0,0,0]

Nae = 437
# Nai=50
Ke = 24
# Ki=397
G = 3.8
ADP = 0.05
Pi = 4.2
ATP = np.exp(G) * ADP * Pi
charge = 1.6 * 10 ** (-19)
NA = 6.02 * 10 ** (23)
RTF = 25.8
R = 238
sigma_pump = 2000
El = -49.4
gl = 0.3
gNa = 120
gK = 36
Cm = 1

popt = np.loadtxt('parameter.txt')[0:14]
[k1p, k1n, k2p, k2n, k3p, k3n, k4p, k4n, KdNae0, KdNai0, KdKe, KdKi, KdATP, delta] = popt.tolist()

dt=0.01
time1=np.arange(0,10,dt)

I=0
results1=odeint(HHmodel,initial,time1,args=([I,delta],))
time2=np.arange(10,60,dt)
I=10
results2=odeint(HHmodel,results1[-1],time2,args=([I,delta],))

time=np.hstack((time1,time2)) #ms
result=np.vstack((results1,results2))
J=[]
for i in range(len(time)):
    v=result[i,0]
    KdNae = KdNae0 * np.exp((1 + delta) * v / (3 * RTF))
    B=result[i,7]
    C=result[i,8]
    Naea = Nae / KdNae
    Kea = Ke / KdKe
    alpha2p = k2p
    alpha2n = k2n * Naea ** 3 / ((1 + Naea) ** 3 + (1 + Kea) ** 2 - 1)
    Jp = (alpha2p * B - alpha2n*C)*1000
    print(alpha2p,B,alpha2n,C,Jp)
    J.append(Jp)

fig=plt.figure()
ax1=fig.add_subplot(211)
ax1.plot(time, J, label='dynamic')
ax1.plot(time, result[:,0])
ax1.set_ylim((-100,100))
plt.legend()
ax2=fig.add_subplot(212)
ax2.plot(time,result[:,6],label='A')
ax2.plot(time,result[:,7],label='B')
ax2.plot(time,result[:,8],label='C')
ax2.plot(time,result[:,9],label='D')
plt.legend()
plt.show()