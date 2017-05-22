function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

pooledFeatures = zeros(convolvedDim / poolDim, ...
        convolvedDim / poolDim, numFilters, numImages);

% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region.
%
%   Use mean pooling here.

%%% YOUR CODE HERE %%%

for imageNum=1:numImages
    for filterNum=1:numFilters

        % Mean pooling of the convolved images
        pooledImage = zeros(convolvedDim / poolDim, convolvedDim / poolDim);

        % Obtain the convolved image
        im = squeeze(convolvedFeatures(:, :, filterNum, imageNum));

        % Convolve the features with unit matrix. Since unit matrix is perfectly
        % symmetrical, no need to perform the flip operations
        % Also, cnvolving means we have already summed up all the pixel intensities
        % In the 3*3 range.
        polI = conv2(im, ones(poolDim), shape='valid');

        % Skip the overlapping convolved images in the X and y direction, so as
        % To obtain only the mutually disjoint features
        polI = polI(1:poolDim:end, 1:poolDim:end);
        polI = bsxfun(@rdivide, polI, poolDim^2);
        pooledFeatures(:, :, filterNum, imageNum) = polI;
    end
end

end
