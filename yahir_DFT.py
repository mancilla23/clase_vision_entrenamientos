import numpy as np

def mi_DFT(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for m in range(N):
        suma = 0
        for n in range(N):
            arg = (2 * np.pi * m * n) / N
            suma += x[n] * np.exp(-1j * arg)
        X[m] = suma
    return X

###otro codigo DFT_funciondiscreta

import numpy as np
from codigoDFT import mi_DFT
import matplotlib.pyplot as plt

t = np.linspace(0,0.001,500)
xt = np.sin( 2*np.pi*1000*t) + 0.5*np.sin(2*np.pi*2000*t + 3/4*np.pi)

fs = 8000
n = np.arange(0,8)
xn = np.sin( 2*np.pi*1000*n/fs) + 0.5*np.sin(2*np.pi*2000*n/fs + 3/4*np.pi)
print(xn)

plt.figure()
plt.plot( t, xt, 'm')
plt.stem( n/fs, xn)
plt.grid()

Xm= mi_DFT(xn)

tol = 1e-14
Xm.real[abs(Xm.real) < tol ] = 0
Xm.imag[abs(Xm.imag) < tol ] = 0
print(Xm)

N = len(xn)
m = np.arange(N)
fan = m*fs/N

plt.figure()
plt.subplot(2,2,1)
plt.stem(fan, Xm.real)
plt.grid()
plt.title('Parte real de la DFT')

plt.subplot(2,2,2)
plt.stem(fan, Xm.imag)
plt.grid()
plt.title('Parte imaginaria de la DFT')

plt.subplot(2,2,3)
plt.stem(fan, np.abs(Xm))
plt.grid()
plt.title('Magnitud de la DFT')

plt.subplot(2,2,4)
plt.stem(fan, np.angle(Xm, deg=True))
plt.grid()
plt.title('Fase de la DFT')
#corregir wato
xtr = (2/N)*(np.sqrt(2)*np.cos(2*np.pi*2000*t) + 4*np.sin(2)*np.cos(2*np.pi*2000*t + 3/4*np.pi))

##otro codigo 2.4
import numpy as np
import matplotlib.pyplot as plt
from codigoDFT import mi_DFT

n = np.arange(11)
F = 1/10
xn = 3*np.sin(2*np.pi*F*n)

plt.figure()
plt.stem(n, xn)
plt.grid()

N = 512
xnr = np.zeros(N)
xnr[0:len(xn)] = xn

Xm = mi_DFT(xnr)

magXm = np.abs(Xm)
m = np.arange(N)*F

plt.figure()
plt.subplot(3,1,1)
plt.plot(m, Xm.imag)
plt.grid()
plt.ylabel('Parte imaginaria de la DFT')

plt.subplot(3,1,2)
plt.plot(m, Xm.real)
plt.grid()
plt.ylabel('Parte real de la DFT')

plt.subplot(3,1,3)
plt.plot(m, magXm)
plt.grid()
plt.ylabel('Magnitud de la DFT')

mm = 5
print( magXm[mm]-magXm[N-mm] )


n = np.arange(11)
F = 1/10
xn = 3*np.exp(2*np.pi*F*n)

plt.figure()
plt.stem(n, xn)
plt.grid()

N = 512
xnr = np.zeros(N, dtype=complex)
xnr[0:len(xn)] = xn

Xm = mi_DFT(xnr)

magXm = np.abs(Xm)
m = np.arange(N)*F

plt.figure()
plt.subplot(3,1,1)
plt.plot(m, Xm.imag)
plt.grid()
plt.ylabel('Parte imaginaria de la DFT')

plt.subplot(3,1,2)
plt.plot(m, Xm.real)
plt.grid()
plt.ylabel('Parte real de la DFT')

plt.subplot(3,1,3)
plt.plot(m, magXm)
plt.grid()
plt.ylabel('Magnitud de la DFT')

mm = 5
print( magXm[mm]-magXm[N-mm] )