function l2reg_logistic()

% load data, plot data and surfaces
[X,y] = load_data();

%%% run grad descent with 2 starting points and plot descent path %%%
lams = [0 0.1];

for q = 1:length(lams)
    % plot pts and surface
    figure(q)
    [s,p,non_obj] = plot_surface(X,y,lams(q));
    

    % run grad descent with first starting point
    w0 = [0;2];
    [in,out] = grad_descent(X,y,w0,lams(q));
    

    % plot results
    plot_pts(X,y);
    hold on
    plot_sigmoid(X,y,in(end,:),1);
    
 
    plot_descent_path(in,out,1,s,p,non_obj);
    
    
%     subplot(1,3,3)
%     [s,p,non_obj] = plot_surface(X,y,lams(q));
%     hold on
%     plot_descent_path(in,out,1,s,p,non_obj);

    % run grad descent with second starting point
    w0 = [0;-2];
    [in,out] = grad_descent(X,y,w0,lams(q));

    % plot results
    plot_sigmoid(X,y,in(end,:),2)
    
    plot_descent_path(in,out,2,s,p,non_obj)
   
    subplot(1,3,1)
    axis([0 max(X(:,2)) 0 1])
    subplot(1,3,3)
    axis([-3 3 -3 3])
    for i = 1:3
        subplot(1,3,i)
        axis square
        box off
    end
end



%%%%%%%%%% subfunctions %%%%%%%%%
% perform grad descent 
function [in,out] = grad_descent(X,y,w,lam)
%     % conservatively optimal fixed step length
%     L = norm(A)^2;
%     alpha = 1/(0.154*L + 2*lam);
    
    % Initializations 
    in = [w];
    d = obj(w);
    out = [d];
    grad = 1;
    iter = 1;
    max_its = 5000;
    
    % main loop
    while  norm(grad) > 10^-5 && iter < max_its

        % fixed steplength
        t = 1./(1 + exp(-X*w));
        grad = 2*(X'*(-t.^3 + (1 + y).*t.^2 - y.*t));
        grad(2) = grad(2) + 2*lam*w(2);
        
        d = obj(w);
        alpha = adaptive_step(w,grad,d);
        w = w - alpha*grad;

        % update containers
        in = [in w];
        out = [out ; d];
        iter = iter + 1;
    end
    in = in';
    
    function p = adaptive_step(z,g,o)
        g_n = norm(g)^2;
        step_l = 1;
        step_r = 0;
        u = 1;
        p = 1;
        while step_l > step_r && u < 10
            p = p*10^-1;
            % left
            step_l = obj(z - p*g) - o;
            
            % right 
            step_r = -(p*g_n)/2;
            u = u + 1;
        end
    end    
    
    function d = obj(s)
         d = norm(1./(1 + exp(-X*s)) - y)^2 + lam*s(2)^2;
    end
end

function [s,t,non_obj] = plot_surface(A,b,lam)
    % setup surface
    range = 3;                     % range over which to view surfaces
    [s,t] = meshgrid(-range:0.2:range);
    s = reshape(s,numel(s),1);
    t = reshape(t,numel(t),1);
    non_obj = zeros(length(s),1);   % nonconvex surface

    % build surface
    for i = 1:length(b)
        non_obj = non_obj + non_convex(A(i,:),b(i),s,t)';
    end
    non_obj = non_obj + lam*t.^2;
    
    % plot surface
    subplot(1,3,2)
    set(gcf,'color','w');
    r = sqrt(numel(s));
    s = reshape(s,r,r);
    t = reshape(t,r,r);
    non_obj = reshape(non_obj,r,r);
    surf(s,t,non_obj)
    box off
    view(-100,10)
    zlabel('g ','Fontsize',18,'FontName','cmr10')
    set(get(gca,'ZLabel'),'Rotation',0)

    % plot contour
    subplot(1,3,3)
    set(gcf,'color','w');
    r = sqrt(numel(s));
    s = reshape(s,r,r);
    t = reshape(t,r,r);
    non_obj = reshape(non_obj,r,r);
    contourf(s,t,non_obj,10)
    view(-90,90)
    box on
    xlabel('w','Fontsize',18,'FontName','cmmi9')
    ylabel('b','Fontsize',18,'FontName','cmmi9')
    set(get(gca,'YLabel'),'Rotation',0)
    
    xlabel('w ','Fontsize',18,'FontName','cmr10')
    ylabel('b ','Fontsize',18,'FontName','cmr10')
    set(get(gca,'YLabel'),'Rotation',0)
    set(get(gca,'XLabel'),'Rotation',0)

    
    function s = non_convex(c,z,s,t)    % objective function for nonconvex problem
         s = (sigmoid(c*[s,t]') - z).^2;
    end
end

function plot_pts(A,b)
    subplot(1,3,1)
    % plot labeled points
    scatter(A(:,2),b,'fill','k')
    set(gcf,'color','w');
    xlabel('x','Fontsize',18,'FontName','cmr10')
    ylabel('y  ','Fontsize',18,'FontName','cmr10')
    set(get(gca,'YLabel'),'Rotation',0)
    set(gcf,'color','w');
    box on
end

function plot_sigmoid(A,b,x,i)
    % plot
    subplot(1,3,1)
    hold on
    u = [0:0.1:max(A(:,2))];
    w = 1./(1+exp(-(u*x(2) + x(1))));
    if i == 1
        plot(u,w,'m','LineWidth',2);
    else
        plot(u,w,'g','LineWidth',2);
    end
end

function plot_descent_path(in,out,i,s,t,non_obj)

    subplot(1,3,2)
    hold on
    if i == 1
       plot3(in(:,1),in(:,2),out,'m','LineWidth',3);
    else
       plot3(in(:,1),in(:,2),out,'g','LineWidth',3);
    end
    axis([min(min(t)) max(max(t)) min(min(s)) max(max(s)) min(min(non_obj)) max(max(non_obj))])
    
    subplot(1,3,3)
    hold on
    if i == 1
       plot3(in(:,1),in(:,2),out,'m','LineWidth',3);
    else
       plot3(in(:,1),in(:,2),out,'g','LineWidth',3);
    end
end

% loads data for processing
function [X,y] = load_data()     
    % load bacteria data
    data = csvread('bacteria_data.csv');
    x = data(:,1);
    y = data(:,2);
    X = [ones(length(x),1) x];
end

function y = sigmoid(z)
y = 1./(1+exp(-z));
end

end
