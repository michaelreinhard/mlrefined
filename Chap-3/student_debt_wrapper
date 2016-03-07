### playing around with syntax for matrix operations
# import all functions directly from numpy, scipy, and matplotlib
import numpy as np
import matplotlib.pylab as plt

# load array
data = np.matrix(np.genfromtxt('student_debt.csv', delimiter=','))
x = data[:,0]
y = data[:,1]

# find smallest LS solution for fit
temp = np.ones((np.size(x),1))
x = np.concatenate((temp,x),1)
X = np.transpose(x)*x
Y = np.transpose(x)*y
w = np.linalg.solve(X,Y)

# plot data
plt.figure(facecolor='white')
ax = plt.plot(x, y,'ko',label = 'student debt')

# plot fit
s = np.arange(min(x[:,1]), max(x[:,1]), 0.1)
t = w.item(0) + s*w.item(1)
plt.plot(s,t,'-r')

# dress up graph
plt.xlabel('year')
plt.ylabel('debt (in trillions of $)')
plt.title('Total U.S. Student Debt')
plt.xlim(min(x[:,1]), max(x[:,1]))
plt.ylim(min(y),max(y))
plt.grid(False)
plt.show()
