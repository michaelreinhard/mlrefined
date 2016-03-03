# convex_newt_surrogate.py is a toy wrapper to illustrate the path
# taken by Hessian descent (or Newton's method).  The steps are evaluated
# at the objective, and then plotted.  For the first 5 iterations the
# quadratic surrogate used to transition from point to point is also plotted.
# The plotted points on the objective turn from green to red as the
# algorithm converges (or reaches a maximum iteration count, preset to 50).
# The (convex) function here is
#
# g(w) = log(1 + exp(w^2))
from numpy import *
from matplotlib.pyplot import *
from pylab import *
import time

def obj(y):
    z = log(1 + exp(y**2))
    return z
def grad(y):
    z = (2*exp(y**2)*y)/(exp(y**2) + 1)
    return z
def hess(y):
    z = (2*exp(y**2)*(2*y**2 + exp(y**2) + 1))/(exp(y**2) + 1)**2
    return z
def surrogate(y,x):
    z = obj(y) + grad(y)*(x - y) + 0.5*hess(y)*(x - y)*(x - y)

    return z

###### ML Algorithm functions ######
def newtons_method(w0):

    #initializations
    grad_stop = 10**-5
    max_its = 50
    iter = 1
    grad_eval = 1
    g_path = []
    w_path = []
    w_path.append(w0)
    g_path.append(obj(w0))
    w = w0
    #main loop
    while linalg.norm(grad_eval) > grad_stop and iter <= max_its:
        #take gradient step
        grad_eval = grad(w)
        hess_eval = hess(w)
        w = w - grad_eval/hess_eval

        #update containers
        w_path.append(w)
        g_path.append(log(1 + exp(w**2)))

        #update stopers
        iter+= 1
    return w,w_path, g_path

###### plotting functions #######
def make_function():
    # plot the function
    global fig,ax1
    fig = figure(facecolor = 'white')
    ax1 = fig.add_subplot(111)
    s = linspace(-2,2,200)
    t = log(1 + exp(s**2))
    ax1.plot(s,t,'-k',linewidth = 2)

    # pretty the figure up
    ax1.set_xlim(-2,2)
    ax1.set_ylim(.5,4)
    ax1.set_xlabel('$w$',fontsize=20,labelpad = 20)
    ax1.set_ylabel('$g(w)$',fontsize=20,rotation = 0,labelpad = 25)

def plot_steps_with_surrogate(w_path,g_path):
    #colors for points
    s = linspace(1/len(g_path),1,len(g_path))
    s.shape = (len(s),1)
    colorspec = concatenate((s,flipud(s)),1)
    colorspec = concatenate((colorspec,zeros((len(s),1))),1)

    #plot initial point
    ax1.plot(w_path[0],g_path[0],'o',markersize = 9, color = colorspec[0,:], markerfacecolor = colorspec[0,:])
    draw()
    # ax1.annotate('$w$'+str(0),(w_path[0],.5))
    t = linspace(.5,g_path[0],100)
    s = w_path[0]*ones((100))
    ax1.plot(s,t,'--k')
    draw()
    time.sleep(2)

    #plot first surrogate and point traveled to
    s_range = 3
    s = linspace(w_path[0]-s_range,w_path[0]+s_range,10000)
    t = surrogate(w_path[0],s)
    h, = ax1.plot(s,t,'--m')
    ind = argmin(t)
    x_mark, = ax1.plot(s[ind],t[ind],'ko',markersize = 6)
    draw()

    for i in range(1,len(g_path)):
            if i <= 2:
                time.sleep(4.5)

                #plot point
                ax1.plot(w_path[i],g_path[i],'o',markersize = 9, color = colorspec[i-1,:], markerfacecolor = colorspec[i-1,:])
                draw()

                # remove old quadratic and stationary pt from drawing
                time.sleep(1)
                h.remove()
                x_mark.remove()
                draw()

                # draw new quadratic and minimum/maximum
                s_range = 3
                s = linspace(w_path[i]-s_range,w_path[i]+s_range,10000)
                t = surrogate(w_path[i],s)
                time.sleep(1)
                h, = ax1.plot(s,t,'--m')
                ind = argmin(t)
                x_mark, = ax1.plot(s[ind],t[ind],'ko',markersize = 6)
                draw()

            if i == 2:
                # remove quadratic and pt
                time.sleep(1)
                h.remove()
                x_mark.remove()
                draw()

            if i >= 2: # just plot point so things don't get too cluttered
                time.sleep(0.01)
                ax1.plot(w_path[i],g_path[i],'o',markersize = 9, color = colorspec[i-1,:], markerfacecolor = colorspec[i-1,:])
                draw()

            # if i == len(g_path) - 1:
            #     h.remove()
            #     x_mark.remove()
            #     time.sleep(1.5)
            #     # ax1.annotate('$w$'+str(i),(w_path[i],.5))
            #     t = linspace(.5,g_path[i],100)
            #     s = w_path[i]*ones((100))
            #     ax1.plot(s,t,'--k')
    show(True)

def main():
    # plot objective function
    make_function()
    pts = matrix(ginput(1))
    x = pts[:,0]
    w0 = float(x[0])            # grab user defined initial pt

    # perform newton's method
    w,w_path,g_path = newtons_method(w0)
    plot_steps_with_surrogate(w_path,g_path)

main()
