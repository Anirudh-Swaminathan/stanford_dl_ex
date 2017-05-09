function [f,g] = linear_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The target value for each example.  y(j) is the target for example j.
  %

  m=size(X,2);
  n=size(X,1);

  f=0;
  g=zeros(size(theta));
  h_x = zeros(size(y));

  %
  % TODO:  Compute the linear regression objective by looping over the examples in X.
  %        Store the objective function value in 'f'.
  %
  % TODO:  Compute the gradient of the objective with respect to theta by looping over
  %        the examples in X and adding up the gradient for each example.  Store the
  %        computed gradient in 'g'.

    for i=1:m
        h_xi = 0;
        for j=1:n
            h_xi = h_xi + theta(j)*X(j,i);
        end
        h_x(i) = h_xi;
        f = f + 0.5*((h_xi - y(i))**2);
    end

    for j=1:n
        for i=1:m
            g(j) = g(j) + (h_x(i) - y(i))*X(j, i);
        end
    end
%%% YOUR CODE HERE %%%
