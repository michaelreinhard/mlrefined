
import numpy as np
import matplotlib.pyplot as plt


# load the data
def load_data():
    # load data
    data = np.matrix(np.genfromtxt('4class_data.csv', delimiter=','))
    x = np.asarray(data[:,0:2])
    temp = np.shape(x)
    temp = np.ones((temp[0],1))
    X = np.concatenate((temp,x),1)
    X = X.T
    y = np.asarray(data[:,2])
    y.shape = (np.size(y),1)

    # return needed variables
    return X,y

###### ML Algorithm functions ######
# learn all C separators
def learn_separators(X,y):
    W = []
    num_classes = np.size(np.unique(y))
    for i in range(0,num_classes):
        # prepare temporary C vs notC probem labels
        y_temp = np.copy(y)
        ind = np.argwhere(y_temp == (i+1))
        ind = ind[:,0]
        ind2 = np.argwhere(y_temp != (i+1))
        ind2 = ind2[:,0]
        y_temp[ind] = 1
        y_temp[ind2] = -1
        # run descent algorithm to classify C vs notC problem
        w = newtons_method(np.random.randn(3,1),X,y_temp)
        W.append(w)
    W = np.asarray(W)
    W.shape = (num_classes,3)
    W = W.T
    return W

# run newton's method
def newtons_method(w0,X,y):
    ## uncomment to record objective value at each iteration to check that algorithm is working
    # obj_path = []
    # obj = calculate_obj(w0,X,y)
    # obj_path.append(obj)
    w = w0

    # start gradient descent loop
    H = np.dot(np.diag(y[:,0]),X.T)
    s = np.shape(y)
    s = s[0]
    l = np.ones((s,1))
    grad = 1
    iter = 1
    max_its = 100
    while np.linalg.norm(grad) > 10**(-5) and iter <= max_its:
        # compute gradient
        temp = 1/(1 + my_exp(np.dot(H,w)))
        grad = - np.dot(H.T,temp)

        # compute Hessian
        g = temp*(l - temp)
        hess = np.dot(np.dot(X,np.diag(g[:,0])),X.T)

        # take Newton step = solve Newton system
        temp = np.dot(hess,w) - grad
        w = np.dot(np.linalg.pinv(hess),temp)
        # w = linalg.solve(hess, dot(hess,w) - grad)

        # update path containers
        # obj = calculate_obj(w,X,y)
        # obj_path.append(obj)
        iter+= 1

    # show final average gradient norm for sanity check
    s = np.dot(grad.T,grad)/np.size(grad)
    s = 'The final average norm of the gradient = ' + str(float(s[0])) + ' in ' + str(iter) + ' iterations of newtons method'
    print(s)

    # # for use in testing if algorithm minimizing/converging properly
    # obj_path = asarray(obj_path)
    # obj_path.shape = (iter,1)
    # plot(asarray(obj_path))
    # show()

    return w

# calculate the objective value for a given input weight w
def calculate_obj(w,X,y):
    obj = np.log(1 + my_exp(-y*np.dot(X.T,w)))
    obj = obj.sum()
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

# calculate the objective value for a given input weight w
def calculate_obj(w):
    obj = np.log(1 + my_exp(-y*np.dot(X.T,w)))
    obj = obj.sum()
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

###### plotting functions #######

def plot_data_and_subproblem_separators(X,y,W):
    # initialize figure, plot data, and dress up panels with axes labels etc.,
    num_classes = np.size(np.unique(y))
    color_opts = np.array([[1,0,0.4], [ 0, 0.4, 1],[0, 1, 0.5],[1, 0.7, 0.5],[0.7, 0.6, 0.5]])
    f,axs = plt.subplots(1,num_classes + 1,facecolor = 'white')

    r = np.linspace(0,1,150)
    for a in range(0,num_classes):
        # color current class
        axs[a].scatter(X[1,],X[2,], s = 30,color = '0.75')
        s = np.argwhere(y == a+1)
        s = s[:,0]
        axs[a].scatter(X[1,s],X[2,s], s = 30,color = color_opts[a,:])
        axs[num_classes].scatter(X[1,s],X[2,s], s = 30,color = color_opts[a,:])

        # draw subproblem separator
        z = -W[0,a]/W[2,a] - W[1,a]/W[2,a]*r
        axs[a].plot(r,z,'-k',linewidth = 2,color = color_opts[a,:])

        # dress panel correctly
        axs[a].set_xlim(0,1)
        axs[a].set_ylim(0,1)
        axs[a].set(aspect = 'equal')
    axs[num_classes].set(aspect = 'equal')

    return axs

# fuse individual subproblem separators into one joint rule
def plot_joint_separator(W,axs,num_classes):
    r = np.linspace(0,1,300)
    s,t = np.meshgrid(r,r)
    s = np.reshape(s,(np.size(s),1))
    t = np.reshape(t,(np.size(t),1))
    h = np.concatenate((np.ones((np.size(s),1)),s,t),1)
    f = np.dot(W.T,h.T)
    z = np.argmax(f,0)
    f.shape = (np.size(f),1)
    s.shape = (np.size(r),np.size(r))
    t.shape = (np.size(r),np.size(r))
    z.shape = (np.size(r),np.size(r))

    for i in range(0,num_classes + 1):
        axs[num_classes].contour(s,t,z,(i + 0.5,i + 0.5),colors = 'k',linewidths = 2.25)

def main():
    # load the data
    X,y = load_data()

    # learn all C vs notC separators
    W = learn_separators(X,y)

    # plot data and each subproblem 2-class separator
    axs = plot_data_and_subproblem_separators(X,y,W)

    # plot fused separator
    plot_joint_separator(W,axs,np.size(np.unique(y)))

    plt.show()

main()
