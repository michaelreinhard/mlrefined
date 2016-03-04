# import all functions directly from numpy, scipy, and matplotlib
from pylab import *

# build functions
x = arange(0.0, 2.0, 0.01)
y = sin(2*pi*x)
y2 = cos(2*pi*x)

# ploting
plot(x, y,'-c',label = 'sine')
plot(x, y2,'-r',label = 'cosine')

# label graph and control axes
xlabel('x')
ylabel('y')
xlim(0,2)
ylim(-2,2)
title('A title')
legend(loc = 'upper right')
grid(True)
show()

# a cool to do feature of pycharm
# todo: this is a comment
