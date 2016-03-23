function softmax_multiclass_demo()
clear all

% load data
red =  [ 1 0 .4];
blue =  [ 0 .4 1];
green = [0 1 0.5];
cyan = [1 0.7 0.5];
grey = [.7 .6 .5];
colors = [red;blue;green;cyan;grey];     % 4 maximum classes
[X,y] = load_data();

% find separators
W = softmax_multiclass_grad(X,y);

% % plot separators one-at-a-time
% figure(1)
% plot_one_ata_time(W,X,y)

% plot the max(..) separator
figure(2)
plot_max_separator(W,X,y)

%%%%%%%%%%%%%%%%% functions %%%%%%%%%%%%%%%%%%%
%%% plot max separator %%%
function plot_max_separator(W,X,y,deg)
    % plot data
    subplot(1,3,1)
    plot_data(X,y)
    subplot(1,3,2)
    plot_data(X,y)
    subplot(1,3,3)
    plot_data(X,y)

    %%% plot all linear separators %%%
    subplot(1,3,2)
    num_classes = length(unique(y));
    x = [0:0.01:1];
    for j = 1:num_classes
        hold on
        w = W(:,j);    
        plot (x,(-w(1)-w(2)*x)/w(3),'Color',colors(j,:),'linewidth',2);
    end

    %%% generate max-separator surface %%%
    s = [0:0.0005:1];
    [s1,s2] = meshgrid(s,s);
    s1 = reshape(s1,numel(s1),1);
    s2 = reshape(s2,numel(s2),1);
    
    % compute criteria for each point in the range [0,1] x [0,1]
    p = [ones(size(s1(:),1),1),s1(:), s2(:)];
    f = W'*p';         
    [f,z] = max(f,[],1);
    
    % fill in appropriate regions with class colors
    co = zeros(size(p,1),3);
    ind = find(z == 1);
    co(ind,:) = repmat(red,length(ind),1);
    ind = find(z == 2);
    co(ind,:) = repmat(blue,length(ind),1);
    ind = find(z == 3);
    co(ind,:) = repmat(green,length(ind),1);
    ind = find(z == 4);   
    co(ind,:) = repmat(cyan,length(ind),1);  
    ind = find(z == 5);
    co(ind,:) = repmat(grey,length(ind),1);  
    hold on

    %%% plot colored region
     %scatter(s1,s2,1,co)
 
    
    % produce decision boundary
    s1 = reshape(s1,[length(s),length(s)]);
    s2 = reshape(s2,[length(s),length(s)]);   
    z = reshape(z,[length(s),length(s)]);   
    num_classes = length(unique(z));
    subplot(1,3,3)
    for i = 1:num_classes - 1
       hold on
       contour(s1,s2,z,[i + 0.5,i + 0.5],'Color','k','LineWidth',2)
    end
    
    % make plot real nice lookin'
    for i = 1:3
        subplot(1,3,i)
        axis([0 1 0 1])
        axis square
        xlabel('x_1','FontName','cmmi9','Fontsize',18)
        ylabel('x_2','FontName','cmmi9','Fontsize',18)
        set(get(gca,'YLabel'),'Rotation',0)
        set(gcf,'color','w');
    end
end

%%% plot learned separators one at a time
function plot_one_ata_time(W,X,y)

    %%% plot one separator at a time %%%
    class_labels = unique(y);           % class labels
    num_classes = length(unique(y));
    x = [0:0.01:1];

    for i = 1:num_classes
        subplot(1,num_classes+1,i)
        w = W(:,i);    
        plot (x,(-w(1)-w(2)*x)/w(3),'Color',colors(i,:),'linewidth',2);

        class = class_labels(i);
        ind = find(y == class);
        hold on
        scatter(X(ind,1),X(ind,2),'Linewidth',2,'Markeredgecolor',colors(i,:),'markerFacecolor','none');
        hold on
        ind = find(y ~= class);
        scatter(X(ind,1),X(ind,2),'Linewidth',2,'Markeredgecolor',[0.5 0.5 0.5],'markerFacecolor','none');
        axis([0 1 0 1])
        axis square
        box on

    end
    
    %%% plot fused separator %%%
    subplot(1,num_classes+1,num_classes+1)
    plot_data(X,y)
    
    % generate max-separator surface 
    s = [0:0.0005:1];
    [s1,s2] = meshgrid(s,s);
    s1 = reshape(s1,numel(s1),1);
    s2 = reshape(s2,numel(s2),1);
    
    % compute criteria for each point in the range [0,1] x [0,1]
    p = [ones(size(s1(:),1),1),s1(:), s2(:)];
    f = W'*p';         
    [f,z] = max(f,[],1);
    
    % fill in appropriate regions with class colors
    co = zeros(size(p,1),3);
    ind = find(z == 1);
    co(ind,:) = repmat(red,length(ind),1);
    ind = find(z == 2);
    co(ind,:) = repmat(blue,length(ind),1);
    ind = find(z == 3);
    co(ind,:) = repmat(green,length(ind),1);
    ind = find(z == 4);
    co(ind,:) = repmat(cyan,length(ind),1);  
    ind = find(z == 5);
    co(ind,:) = repmat(grey,length(ind),1);  
    hold on
    
    % produce decision boundary
    s1 = reshape(s1,[length(s),length(s)]);
    s2 = reshape(s2,[length(s),length(s)]);   
    z = reshape(z,[length(s),length(s)]);   
    num_classes = length(unique(z));
    for i = 1:num_classes - 1
       hold on
       contour(s1,s2,z,[i + 0.5,i + 0.5],'Color','k','LineWidth',2)
    end

    % make subplots real nice lookin'
    for i = 1:num_classes + 1
        subplot(1,num_classes + 1,i)
        axis([0 1 0 1])
        axis square
        xlabel('x_1','FontName','cmmi9','Fontsize',18)
        ylabel('x_2','FontName','cmmi9','Fontsize',18)
        set(get(gca,'YLabel'),'Rotation',0)
    end
    set(gcf,'color','w');
end

%%% gradient descent for softmax multiclass %%%
function W = softmax_multiclass_grad(X,y)
    % initialize
    X = [ones(size(X,1),1), X]';
    [N,P] = size(X);
    C = length(unique(y));
    W = randn(N,C);
    
    % precomputations
    grad = 1;
    t = 1;
    l = ones(P,1);
    
    %%% main %%%
    while t <= 5000 && norm(grad) > 10^-6
        % prep gradient
        s = exp(X'*W);
        grad = zeros(size(W));
        q = sum(s,2);
        for c = 1:C
            u = s(:,c)./q;
            z = l;
            ind = find(y ~= c);
            z(ind) = 0;
            U = diag(u  - z);
            grad(:,c) = X*U*l;
            b = 1;
        end
        alpha = adaptive_step(W,grad,obj(X,W),X);
        W = W - alpha*grad;
        
        % update counter
        t = t + 1;
    end
end

%%% adaptively choose step length %%%
function p = adaptive_step(W,g,o,Z)
    g_n = norm(g,'fro')^2;
    step_l = 1;
    step_r = 0;
    u = 1;
    p = 1;
    while step_l > step_r && u < 10
        p = p*10^-1;

        % left
        step_l = obj(Z,W - p*g) - o;

        % right 
        step_r = -(p*g_n)/2;
        u = u + 1;
    end
end

%%% compute softmax multiclass loss objective %%%
function o = obj(Z,E)
    s = exp(Z'*E);
    q = sum(s,2);
    r = ones(size(s,1),1);
    o = 0;
    for c = 1:length(unique(y))
        u = s(:,c)./q;
        ind = find(y ~= c);
        z = r;
        z(ind) = 0;
        u = u.*z;
        u(ind) = [];
        o = o - sum(log(u));
    end
end

%%% plot full dataset %%%
function plot_data(X,y)
    % how many classes in the data?  4 maximum for this toy.
    class_labels = unique(y);           % class labels
    num_classes = length(class_labels);

    % plot data
    for i = 1:num_classes
        class = class_labels(i);
        ind = find(y == class);
        hold on
        scatter(X(ind,1),X(ind,2),'Linewidth',2,'Markeredgecolor',colors(i,:),'markerFacecolor','none');

        hold on
        scatter(X(ind,1),X(ind,2),'Linewidth',2,'Markeredgecolor',colors(i,:),'markerFacecolor','none');
    end
    axis([0 1 0 1])
    axis square
    box on
end

%%% load data %%%
function [X,y] = load_data()
    % load data from file
    data = csvread('5class_data.csv');
    X = data(:,1:end - 1);
    y = data(:,end);
end
end
