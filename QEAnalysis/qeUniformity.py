import numpy as np
import matplotlib.pylab as plt

#plt.rc('text', usetex=True)
plt.rc('font', family='serif')

diams = np.array([1.75,1,0.5,0.386,0.25])
pows = np.array([135.456, 68.9768, 44.5958, 42.1765, 37.9167])
errs = np.array([0.97417, 1.48325, 2.07886, 2.10242, 1.90503])

dark = 36.1867
darkErr = 1.42287

pows-=dark

units = ['Aperture Area (sq. inches)','Photodiode Measured Power (pW)']

areas = np.power((diams/2.0),2)*np.pi

fig = plt.figure()
plt.errorbar(areas,pows,yerr=3.0*np.sqrt(np.power(errs,2)+np.power(darkErr,2)))
plt.xlabel(units[0],fontsize=14)
plt.ylabel(units[1],fontsize=14)
plt.show()

fig = plt.figure()
plt.errorbar(diams/2,pows/areas, yerr=3.0*np.sqrt(np.power(errs,2)+np.power(darkErr,2)))
plt.xlabel("Field Radius [in]")
plt.ylabel("pW/area")
plt.show()
