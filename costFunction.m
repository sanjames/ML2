function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.
% Initialize some useful values
m = length(y); % number of training examples
m;

J = 0;
grad = zeros(size(theta));

% You need to return the following variables correctly 
%J = 0;
%grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
% vectorized implementation of Cost function is as follows:
%
% J(Theta) = -1/m * ((y transpose * log(h)) + (1-y) transpose * log(1-h)) 
% where h = sigmoid( X, theta)
% sigmoid(z) = 1/(1 + exp(-z))
%

%define h
%h = sigmoid(X * theta); %h is 1 X 100 vector 
% y is 100 X 1 matrix. Y' is 1  X 100 vector
%A = y*log(hx);  
%B = (1-y) * log(1-hx)
%J = 1/(2*m) * (sum((X * theta - y) .^ 2))
%J = () + ((1-y)' * log(1-h)) ;
%J = ((-y')*log(h)- (1-y)' * log(1-h))/m
% =============================================================

%h = sigmoid(X*theta);
h = sigmoid(X*theta);
% size(h) size(y) size(X) pause();
J = ((-1.*y')*log(h)-(1-y)'*log(1-h))/m;

% calculate grads
grad = (X'*(h - y))/m;
J
grad

end
