import numpy as np
import scipy.integrate

data = np.genfromtxt('population.csv', delimiter=',')
t = data[0, :]
N = data[1, :]


########## Problem 1 ###########
dx = t[1] - t[0]
A1 = (3 * N[-1] - 4 * N[-2] + N[-3]) / (2 * dx)
print("A1 = ", A1)

A2 = (N[10] - N[8]) / (2 * dx)
print("A2 = ", A2)

A3 = (-3 * N[0] + 4 * N[1] - N[2]) / (2 * dx)
print("A3 = ", A3)

n = t.size
deriv = np.zeros(n)
deriv[0] = A3
for k in range(1, n - 1):
    deriv[k] = (N[k + 1] - N[k - 1]) / (2 * dx)
deriv[-1] = A1
A4 = deriv.reshape(1,24)
print("A4 = ", A4)

capita = np.zeros(n)
ans = np.zeros(n)
capita[0] = deriv[0] / N[0]
for k in range(1, n - 1):
    capita[k] = deriv[k] / N[k]
capita[-1] = deriv[-1] / N[-1]
A5 = capita.reshape(1,24)
#print("A5 = ", A5)


div = capita.size
avg = np.zeros(div)
avg = capita / div
Old_A6 = avg.reshape(1, 24)
A6 = np.sum(capita) / div
print("A6 = ", A6)

########### Problem 2 ##########
data = np.genfromtxt('brake_pad.csv', delimiter=',')
r = data[0, :]
T = data[1, :]
ro = 0.308
re = 0.478
theta = 0.7051

dx = r[1] - r[0]
y = T * theta * r
LHR = dx * np.sum(y[:-1])
A7 = LHR
#print("A7 = ", LHR)
A = r * theta
LHRA = dx * np.sum(A[:-1])
TBar = LHR/ LHRA
A8 = TBar
#print("A8 = ", TBar)


y = T * theta * r
RHR = dx * np.sum(y[1:])
A9 = RHR
#print("A9 = ", RHR)
RHRA = dx * np.sum(A[1:])
TBar = RHR / RHRA
A10 = TBar
#print("A10 = ", TBar)

y = T * theta * r
A11 = np.trapz(y, r)
#print("A11 = ", A11)
A = r * theta
A = np.trapz(A, r)
A12 = A11 / A
#print('A12 = ', A12)


########## Problem 3 ###########
F = lambda x: ((x**2)/2) - ((x**3)/3)
def reactionDiffusion(u, z):
    ans = u / (np.sqrt(F(u) - F(u * z)))
    return ans

a = 0
b = 1
fA = lambda z: reactionDiffusion(0.95, z)
A13, err = scipy.integrate.quad(fA, a, b)
#print("A13 = ", A13)

fB = lambda z: reactionDiffusion(0.5, z)
A14, err = scipy.integrate.quad(fB, a, b)
#print("A14 = ", A14)

fC = lambda z: reactionDiffusion(0.01, z)
A15, err = scipy.integrate.quad(fC, a, b)
#print("A15 = ", A15)






