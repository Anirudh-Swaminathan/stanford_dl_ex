function numgrad = computeNumericalGradient(J, theta, varargin)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta.

% Initialize numgrad with zeros
numgrad = zeros(size(theta));

%% ---------- YOUR CODE HERE --------------------------------------
% Instructions:
% Implement numerical gradient checking, and return the result in numgrad.
% (See Section 2.3 of the lecture notes.)
% You should write code so that numgrad(i) is (the numerical approximation to) the
% partial derivative of J with respect to the i-th input argument, evaluated at theta.
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with
% respect to theta(i).
%
% Hint: You will probably want to compute the elements of numgrad one at a time.
disp('Gradient Checking');
EPSILON=10^-4;
for j=1:size(theta)
    T = theta;
    T0 = T; T0(j) = T0(j)-EPSILON;
    T1 = T; T1(j) = T1(j)+EPSILON;
    f0 = J(T0, varargin{:});
    f1 = J(T1, varargin{:});
    numgrad(j) = (f1-f0) / (2.0*EPSILON);
end






%% ---------------------------------------------------------------
end
