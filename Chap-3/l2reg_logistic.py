
# This file pairs with chapter 3 of the textbook "Machine Learning Refined" published by Cambridge University Press.  

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# load the data
def load_data():
    # load data
    data = np.matrix(np.genfromtxt('bacteria_data.csv', delimiter=','))
    x = np.asarray(data[:,0])
    y = np.asarray(data[:,1])
    return x,y

###### ML Algorithm functions ######
# run gradient descent
def gradient_descent(x,y,w0,lam):
    # formulate data matrix
    temp = np.ones((np.size(x),1))
    X = np.concatenate((temp,x),1)

    # containers for visualization of path
    w_path = []         # container for weights learned at each iteration
    obj_path = []       # container for associated objective values at each iteration
    w_path.append(w0)
    obj = calculate_obj(X,y,w0,lam)
    obj_path.append(obj)
    grad_path = []
    w = w0

    # start gradient descent loop
    grad = 1
    iter = 1
    max_its = 5000
    alpha = 10**(-2)
    while np.linalg.norm(grad) > 10**(-5) and iter <= max_its:
        # compute gradient
        temp = 1/(1 + my_exp(-np.dot(X,w)))
        grad = 2*np.dot(X.T,-temp**3 + (1+y)*(temp**2) -y*temp)
        grad[1] = grad[1] + 2*lam*w[1]
        # grad_path.append(dot(grad.T,grad))

        # take gradient step
        w = w - alpha*grad

        # update path containers
        w_path.append(w)
        obj = calculate_obj(X,y,w,lam)
        obj_path.append(obj)
        iter+= 1

    # reshape containers for use in plotting in 3d
    w_path = np.asarray(w_path)
    w_path.shape = (iter,2)
    obj_path = np.asarray(obj_path)
    obj_path.shape = (iter,1)

    return (w_path,obj_path)

    ## for use in testing if gradient vanishes
    # grad_path = asarray(grad_path)
    # grad_path.shape = (iter,1)
    # plot(asarray(grad_path))
    # show()

# calculate the objective value for a given input weight w
def calculate_obj(X,y,w,lam):
    temp = 1/(1 + my_exp(-np.dot(X,w))) - y
    temp = np.dot(temp.T,temp)
    obj = temp + lam*w[1]**2
    return obj

# avoid overflow when using exp - just cutoff after arguments get too large/small
def my_exp(u):
    s = np.argwhere(u > 100)
    t = np.argwhere(u < -100)
    u[s] = 0
    u[t] = 0
    u = np.exp(u)
    u[t] = 1
    return u

### plotting functions ###
# make 3d surface plot
def plot_logistic_surface(x,y,lam,ax2):
    # formulate data matrix
    temp = np.ones((np.size(x),1))
    X = np.concatenate((temp,x),1)

    # make a meshgrid for the 3d surface
    r = np.linspace(-3,3,150)
    s,t = np.meshgrid(r, r)
    s = np.reshape(s,(np.size(s),1))
    t = np.reshape(t,(np.size(t),1))
    h = np.concatenate((s,t),1)

    # build 3d surface
    surf = np.zeros((np.size(s),1))
    max_its = np.size(y)
    for i in range(0,max_its):
        surf = surf + add_layer(X[i,:],y[i],h)
    surf = surf + lam*t**2

    s = np.reshape(s,(np.sqrt(np.size(s)),np.sqrt(np.size(s))))
    t = np.reshape(t,(np.sqrt(np.size(t)),np.sqrt(np.size(t))))
    surf = np.reshape(surf,(np.sqrt(np.size(surf)),np.sqrt(np.size(surf))))

    # plot 3d surface
    ax2.plot_surface(s,t,surf,cmap = 'jet')
    ax2.azim = 175
    ax2.elev = 20

# used by plot_logistic_surface to make objective surface of logistic regression cost function
def add_layer(a,b,c):
    a.shape = (2,1)
    b.shape = (1,1)
    z = my_exp(-np.dot(c,a))
    z = 1/(1 + z) - b
    z = z**2
    return z

# plot fit to data and corresponding gradient descent path onto the logistic regression objective surface
def show_fit_and_paths(x,y,ax1,ax2,w_path,obj_path,col):

    # plot solution of gradient descent fit to original data
    s = np.linspace(0,x.max(),100)
    t = 1/(1 + my_exp(-(w_path[-1,0] + w_path[-1,1]*s)))
    ax1.plot(s,t,color = col)

    # plot grad descent path onto surface
    ax2.plot(w_path[:,0],w_path[:,1],obj_path[:,0],color = col,linewidth = 5)   # add a little to output path so its visible on top of the surface plot

# plot data
def plot_data(x,y):
    # initialize figure, plot data, and dress up panels with axes labels etc.,
    fig = plt.figure(facecolor = 'white')
    ax1 = fig.add_subplot(121)
    ax1.set_xlabel('$x$',fontsize=20,labelpad = 20)
    ax1.set_ylabel('$y$',fontsize=20,rotation = 0,labelpad = 20)
    plt.plot(x,y,'ko')
    ax1.set_xlim(min(x[:,0])-0.5, max(x[:,0])+0.5)
    ax1.set_ylim(min(y)-0.1,max(y)+0.1)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.xaxis.set_rotate_label(False)
    ax2.yaxis.set_rotate_label(False)
    ax2.zaxis.set_rotate_label(False)
    ax2.get_xaxis().set_ticks([-3,-1,1,3])
    ax2.get_yaxis().set_ticks([-3,-1,1,3])
    ax2.set_xlabel('$w_0$   ',fontsize=20,rotation = 0,linespacing = 10)
    ax2.set_ylabel('$w_1$',fontsize=20,rotation = 0,labelpad = 50)
    ax2.set_zlabel('   $g(\mathbf{w})$',fontsize=20,rotation = 0,labelpad = 20)

    return ax1,ax2

def main():
    lam = 10**-1              # L2 regularizer parameter, convexifies the objective so gradient descent doesn't get stuck in flat areas as much
    x,y = load_data()               # load the data

    # plot the initials - the data and logistic cost surface
    ax1,ax2 = plot_data(x,y)
    plot_logistic_surface(x,y,lam,ax2)   # plot objective surface

    ### run gradient descent with first initial point
    w0 = np.array([0,2])
    w0.shape = (2,1)
    w_path, obj_path = gradient_descent(x,y,w0,lam)

    # plot fit to data and path on objective surface
    show_fit_and_paths(x,y,ax1,ax2,w_path, obj_path,'m')

    ### run gradient descent with first initial point
    w0 = np.array([0,-2])
    w0.shape = (2,1)
    w_path, obj_path = gradient_descent(x,y,w0,lam)

    # plot fit to data and path on objective surface
    show_fit_and_paths(x,y,ax1,ax2,w_path, obj_path,'c')
    plt.show()

main()
