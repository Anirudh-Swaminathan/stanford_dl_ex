% runs training procedure for supervised multilayer network
% softmax output layer with cross entropy loss function

%% setup environment
% experiment information
% a struct containing network layer sizes etc
ei = [];

% add common directory to your path for
% minfunc and mnist data helpers
addpath ../common;
addpath(genpath('../common/minFunc_2012/minFunc'));

%% load mnist data
[data_train, labels_train, data_test, labels_test] = load_preprocess_mnist();
size(data_train)
size(labels_train)
size(data_test)
size(labels_test)

fprintf('\n\n\n****A total of 12 different networks will be trained and tested for the same MNIST datasets****\n\n');

%% populate ei with the network architecture to train
% ei is a structure you can use to store hyperparameters of the network
% the architecture specified below should produce  100% training accuracy
% You should be able to try different network architectures by changing ei
% only (no changes to the objective function code)

% dimension of input features
ei.input_dim = 784;
% number of output classes
ei.output_dim = 10;
% sizes of all hidden layers and the output layer
ei.layer_sizes = [256, ei.output_dim];
% scaling parameter for l2 weight regularization penalty
ei.lambda = 0;
% which type of activation function to use in hidden layers
% feel free to implement support for only the logistic sigmoid function
ei.activation_fun = 'logistic';

%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% setup minfunc options
options = [];
options.display = 'iter';
options.maxFunEvals = 1e6;
options.Method = 'lbfgs';
options.useMex = 0;

%% run training
tic;
[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
    params,options,ei, data_train, labels_train);
fprintf('Optimization took %f seconds.\n', toc);

%% compute accuracy on the test and train set
[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
[~,pred] = max(pred);
acc_test = mean(pred'==labels_test);
fprintf('test accuracy: %f%%\n', 100.0*acc_test);

[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
[~,pred] = max(pred);
acc_train = mean(pred'==labels_train);
fprintf('train accuracy: %f%%\n', 100.0*acc_train);


% Train with the tanh activation function for the same network
fprintf('\n\nThe 3 layer network having 784-256-10 layers is trained using tanh units\n\n');
ei.activation_fun = 'tanh';
%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% run training
tic;
[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
    params,options,ei, data_train, labels_train);
fprintf('Optimization took %f seconds.\n', toc);

%% compute accuracy on the test and train set
[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
[~,pred] = max(pred);
acc_test = mean(pred'==labels_test);
fprintf('test accuracy: %f%%\n', 100.0*acc_test);

[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
[~,pred] = max(pred);
acc_train = mean(pred'==labels_train);
fprintf('train accuracy: %f%%\n', 100.0*acc_train);


% Train with the reLU activation function for the same network
fprintf('\n\nThe 3 layer network having 784-256-10 layers is trained using reLU\n\n');
ei.activation_fun = 'reLU';
%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% run training
tic;
[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
    params,options,ei, data_train, labels_train);
fprintf('Optimization took %f seconds.\n', toc);

%% compute accuracy on the test and train set
[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
[~,pred] = max(pred);
acc_test = mean(pred'==labels_test);
fprintf('test accuracy: %f%%\n', 100.0*acc_test);

[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
[~,pred] = max(pred);
acc_train = mean(pred'==labels_train);
fprintf('train accuracy: %f%%\n', 100.0*acc_train);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% NETWORK with lambda as 0.12 %%%%
ei.lambda = 0.12;
% Train with the sigmoid activation function for the same network
fprintf('\n\nThe 3 layer network having 784-256-10 layers is trained using sigmoid & lambda=0.2\n\n');
ei.activation_fun = 'logistic';
%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% run training
tic;
[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
    params,options,ei, data_train, labels_train);
fprintf('Optimization took %f seconds.\n', toc);

%% compute accuracy on the test and train set
[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
[~,pred] = max(pred);
acc_test = mean(pred'==labels_test);
fprintf('test accuracy: %f%%\n', 100.0*acc_test);

[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
[~,pred] = max(pred);
acc_train = mean(pred'==labels_train);
fprintf('train accuracy: %f%%\n', 100.0*acc_train);


% Train with the tanh activation function for the same network
fprintf('\n\nThe 3 layer network having 784-256-10 layers is trained using tanh & lambda=0.1\n\n');
ei.activation_fun = 'tanh';
%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% run training
tic;
[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
    params,options,ei, data_train, labels_train);
fprintf('Optimization took %f seconds.\n', toc);

%% compute accuracy on the test and train set
[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
[~,pred] = max(pred);
acc_test = mean(pred'==labels_test);
fprintf('test accuracy: %f%%\n', 100.0*acc_test);

[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
[~,pred] = max(pred);
acc_train = mean(pred'==labels_train);
fprintf('train accuracy: %f%%\n', 100.0*acc_train);


% Train with the reLU activation function for the same network
fprintf('\n\nThe 3 layer network having 784-256-10 layers is trained using reLU & lambda=0.1\n\n');
ei.activation_fun = 'reLU';
%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% run training
tic;
[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
    params,options,ei, data_train, labels_train);
fprintf('Optimization took %f seconds.\n', toc);

%% compute accuracy on the test and train set
[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
[~,pred] = max(pred);
acc_test = mean(pred'==labels_test);
fprintf('test accuracy: %f%%\n', 100.0*acc_test);

[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
[~,pred] = max(pred);
acc_train = mean(pred'==labels_train);
fprintf('train accuracy: %f%%\n', 100.0*acc_train);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% NETWORK with config as 784*128*32*10 and lambda=0 %%%%
ei.layer_sizes = [128, 32, ei.output_dim];
ei.lambda = 0;
% Train with the logistic activation function for the same network
fprintf('\n\nThe 4 layer network having 784-128-32-10 layers is trained using sigmoid units\n\n');
ei.activation_fun = 'logistic';

%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% run training
tic;
[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
    params,options,ei, data_train, labels_train);
fprintf('Optimization took %f seconds.\n', toc);

%% compute accuracy on the test and train set
[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
[~,pred] = max(pred);
acc_test = mean(pred'==labels_test);
fprintf('test accuracy: %f%%\n', 100.0*acc_test);

[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
[~,pred] = max(pred);
acc_train = mean(pred'==labels_train);
fprintf('train accuracy: %f%%\n', 100.0*acc_train);


% Train with the tanh activation function for the same network
fprintf('\n\nThe 4 layer network having 784-128-32-10 layers is trained using tanh units\n\n');
ei.activation_fun = 'tanh';
%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% run training
tic;
[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
    params,options,ei, data_train, labels_train);
fprintf('Optimization took %f seconds.\n', toc);

%% compute accuracy on the test and train set
[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
[~,pred] = max(pred);
acc_test = mean(pred'==labels_test);
fprintf('test accuracy: %f%%\n', 100.0*acc_test);

[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
[~,pred] = max(pred);
acc_train = mean(pred'==labels_train);
fprintf('train accuracy: %f%%\n', 100.0*acc_train);


% Train with the reLU activation function for the same network
fprintf('\n\nThe 4 layer network having 784-128-32-10 layers is trained using reLU\n\n');
ei.activation_fun = 'reLU';
%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% run training
tic;
[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
    params,options,ei, data_train, labels_train);
fprintf('Optimization took %f seconds.\n', toc);

%% compute accuracy on the test and train set
[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
[~,pred] = max(pred);
acc_test = mean(pred'==labels_test);
fprintf('test accuracy: %f%%\n', 100.0*acc_test);

[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
[~,pred] = max(pred);
acc_train = mean(pred'==labels_train);
fprintf('train accuracy: %f%%\n', 100.0*acc_train);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% NETWORK with lambda as 0.12 %%%%
ei.lambda = 0.12;
% Train with the sigmoid activation function for the same network
fprintf('\n\nThe 4 layer network having 784-128-32-10 layers is trained using sigmoid & lambda=0.1\n\n');
ei.activation_fun = 'logistic';
%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% run training
tic;
[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
    params,options,ei, data_train, labels_train);
fprintf('Optimization took %f seconds.\n', toc);

%% compute accuracy on the test and train set
[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
[~,pred] = max(pred);
acc_test = mean(pred'==labels_test);
fprintf('test accuracy: %f%%\n', 100.0*acc_test);

[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
[~,pred] = max(pred);
acc_train = mean(pred'==labels_train);
fprintf('train accuracy: %f%%\n', 100.0*acc_train);


% Train with the tanh activation function for the same network
fprintf('\n\nThe 4 layer network having 784-128-32-10 layers is trained using tanh & lambda=0.1\n\n');
ei.activation_fun = 'tanh';
%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% run training
tic;
[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
    params,options,ei, data_train, labels_train);
fprintf('Optimization took %f seconds.\n', toc);

%% compute accuracy on the test and train set
[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
[~,pred] = max(pred);
acc_test = mean(pred'==labels_test);
fprintf('test accuracy: %f%%\n', 100.0*acc_test);

[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
[~,pred] = max(pred);
acc_train = mean(pred'==labels_train);
fprintf('train accuracy: %f%%\n', 100.0*acc_train);


% Train with the reLU activation function for the same network
fprintf('\n\nThe 4 layer network having 784-128-32-10 layers is trained using reLU & lambda=0.1\n\n');
ei.activation_fun = 'reLU';
%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% run training
tic;
[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
    params,options,ei, data_train, labels_train);
fprintf('Optimization took %f seconds.\n', toc);

%% compute accuracy on the test and train set
[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
[~,pred] = max(pred);
acc_test = mean(pred'==labels_test);
fprintf('test accuracy: %f%%\n', 100.0*acc_test);

[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
[~,pred] = max(pred);
acc_train = mean(pred'==labels_train);
fprintf('train accuracy: %f%%\n', 100.0*acc_train);
