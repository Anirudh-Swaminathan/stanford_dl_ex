function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
size(stack{1}.W);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
%% forward prop
%%% YOUR CODE HERE %%%

for l=1:length(hAct)
    if l==1
        xh = bsxfun(@plus, stack{l}.W*data, stack{l}.b);
        switch ei.activation_fun
        case 'logistic'
            hAct{l} = sigmoid(xh);
        case 'tanh'
            hAct{l} = tanh(xh);
        case 'reLU'
            hAct{l} = bsxfun(@max, zeros(size(xh)), xh);
        end
    elseif l==length(hAct)
        Hwb = exp(bsxfun(@plus, stack{l}.W*hAct{l-1}, stack{l}.b));
        hAct{end} = bsxfun(@rdivide, Hwb, sum(Hwb));
        clear Hwb;
    else
        xh = bsxfun(@plus, stack{l}.W*hAct{l-1}, stack{l}.b);
        switch ei.activation_fun
        case 'logistic'
            hAct{l} = sigmoid(xh);
        case 'tanh'
            hAct{l} = tanh(xh);
        case 'reLU'
            hAct{l} = bsxfun(@max, zeros(size(xh)), xh);
        end
    end
end
pred_prob = hAct{end};

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%

ty = bsxfun(@eq, labels(:), 1:max(labels));
ty = ty';
m = size(data, 2);
crossEntropyCost = -1.0 * sum(sum(ty .* log(pred_prob))) / m;

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%

sDel = cell(numHidden+1, 1);
for i=length(sDel):-1:1
    if i==length(sDel)
        sDel{i} = -(ty - hAct{i});
    elseif i==1
        znl = bsxfun(@plus, stack{i}.W*data, stack{i}.b);
        switch ei.activation_fun
        case 'logistic'
            der = bsxfun(@times, sigmoid(znl), (1-sigmoid(znl)));
        case 'tanh'
            der = bsxfun(@power, sech(znl), 2);
        case 'reLU'
            der = bsxfun(@max, zeros(size(znl)), znl>0);
        end
        sDel{i} = (stack{i+1}.W' * sDel{i+1}) .* der;
    else
        znl = bsxfun(@plus, stack{i}.W*hAct{i-1}, stack{i}.b);
        switch ei.activation_fun
        case 'logistic'
            der = bsxfun(@times, sigmoid(znl), (1-sigmoid(znl)));
        case 'tanh'
            der = bsxfun(@power, sech(znl), 2);
        case 'reLU'
            der = bsxfun(@max, zeros(size(znl)), znl>0);
        end
        sDel{i} = (stack{i+1}.W' * sDel{i+1}) .* der;
    end
end

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

regCost = 0;
for i=1:numel(stack)
    regCost = regCost + (ei.lambda / 2) * norm(stack{i}.W, 'fro')^2;
end
cost = crossEntropyCost + regCost;

% Gradient computations
for i=1:length(gradStack)
    if i==1
        gradStack{i}.W = 1.0 / m * (sDel{i} * data') + ei.lambda * stack{i}.W;
        gradStack{i}.b = 1.0 / m * sum(sDel{i}, 2);
    else
        gradStack{i}.W = 1.0 / m * (sDel{i} * hAct{i-1}') + ei.lambda * stack{i}.W;
        gradStack{i}.b = 1.0 / m * sum(sDel{i}, 2);
    end
end

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end
