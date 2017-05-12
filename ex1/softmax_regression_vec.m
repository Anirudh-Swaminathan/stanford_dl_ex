function [f,g] = softmax_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;

  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
    % Perform 1 hot encoding initially
    ty = bsxfun(@eq, y(:), 1:max(y));
    ty = ty';

    % Calculate the hypothesis
    epow = exp(theta' * X);
    epow = [epow; ones(1, m)];
    h_x = bsxfun(@rdivide, epow, sum(epow));

    f = sum(sum(ty .* log(h_x)));
    f = -1.0 * f;

    g = -1.0 * X * (ty - epow)';

    % Since g is a nxk matrix, we needn't have a theta to represent the last class
    % Hence, we delete the last class g by removing the last column of g
    % to make it a nx(k-1) matrix
    g(:, end) = [];

  g=g(:); % make gradient a vector for minFunc
