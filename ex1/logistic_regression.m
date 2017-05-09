function [f,g] = logistic_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %

  m=size(X,2);
  n = size(X,1);

  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));
  h_x = zeros(size(y));


  %
  % TODO:  Compute the objective function by looping over the dataset and summing
  %        up the objective values for each example.  Store the result in 'f'.
  %
  % TODO:  Compute the gradient of the objective by looping over the dataset and summing
  %        up the gradients (df/dtheta) for each example. Store the result in 'g'.
  %
    for i=1:m
        t_xi = 0;
        for j=1:n
            t_xi = t_xi + theta(j)*X(j,i);
        end
        h_x(i) = sigmoid(t_xi);
        f = f - ((y(i)*log(h_x(i))) + ((1-y(i))*log(1-h_x(i))));
    end
%
    for j=1:n
        for i=1:m
            g(j) = g(j) + (h_x(i) - y(i))*X(j, i);
        end
    end
%%% YOUR CODE HERE %%%
