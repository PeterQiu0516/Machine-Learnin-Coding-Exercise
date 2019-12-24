function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    %qcy: 1.sqrt of cost function in columns: (X*theta-y) m*1 matrix
    %2.take its transpose and times x:  1*m matrix times m*n+1 matrix = 1*n+1 matrix
    %3.take the transpose of the last result and divide by m finally gives the delta 

    %
    %delta = ((X*theta-y)'*X)'/m

    %ssx: 1.sqrt of cost function in columns: (X*theta-y) m*1 matrix
    %2.take the transpose of X, gives n+1*m matrix with rows representing "m" data for a specific parameter
    %3.calculate X' * cost function(X*theta-y) gives a n+1*1 matrix, divide it by m and we obtain delta   
    
    
    delta = X' * (X*theta-y)/m                                                              
    theta = theta - alpha*delta





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
