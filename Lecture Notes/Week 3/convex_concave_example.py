import numpy as np
import matplotlib.pyplot as plt



x = np.linspace(-10,10,100)

fig = plt.figure()
plt.plot(x,x**2,linewidth=3.0)

x1 = 5
x2 = np.linspace(x1-2,x1+2,100)
plt.plot(x2, x1**2 + 2*x1*(x2-x1),'r',linewidth=3.0)

x1 = 0
x2 = np.linspace(x1-2,x1+2,100)
plt.plot(x2, x1**2 + 2*x1*(x2-x1),'r',linewidth=3.0)

x1 = -5
x2 = np.linspace(x1-2,x1+2,100)
plt.plot(x2, x1**2 + 2*x1*(x2-x1),'r',linewidth=3.0)

plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.title('Convex: $f(x)=x^2$')

fig.savefig('convex_fnc.png')



x = np.linspace(-np.pi/2,np.pi/2,100)

fig = plt.figure()
plt.plot(x,np.cos(x),linewidth=3.0)

x1 = -1
x2 = np.linspace(x1-0.45,x1+0.45,100)
plt.plot(x2, np.cos(x1) - np.sin(x1)*(x2-x1),'r',linewidth=3.0)

x1 = 0
x2 = np.linspace(x1-0.5,x1+0.5,100)
plt.plot(x2, np.cos(x1) - np.sin(x1)*(x2-x1),'r',linewidth=3.0)

x1 = 1
x2 = np.linspace(x1-0.45,x1+0.45,100)
plt.plot(x2, np.cos(x1) - np.sin(x1)*(x2-x1),'r',linewidth=3.0)


plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.title('Concave: $f(x)=\cos(x)$')

fig.savefig('concave_fnc.png')



plt.show()