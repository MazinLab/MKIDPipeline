from scipy import special
import numpy as np
import matplotlib.pylab as plt
plt.rc('font',family='serif')

#airy = A * [2 * J1 (pi*r/(R/Rz) / (pi*r/(R/Rz))]^2

r = np.arange(-50,50)
Rz = 1.21966989
R=1.22e-7
pi = np.pi
A=1.

denom = pi*r/(R/Rz)
bess = special.jv(1,denom)
airy = np.power(2.*bess/denom,2.)

airy[np.where(r==0)]=A

plt.plot(r,airy)
plt.yscale('log')
plt.show()

