%% data load
clear
load('DigitData.mat','imset_im','classification')
load('mnist_patternnet_16h.mat','network')

train_network = 0;

if train_network
    subset = imset_im(1,1:2000);
    subset_class = classification(1,1:2000);

    subset_vec = zeros(784, 2000);

    for a=1:2000
        subset_vec(:, a) = reshape(subset{a},1,784);
    end

    hot = full(ind2vec(double(subset_class),10));
    hot = circshift(hot,[-1 0]);

    training = subset(1,1:1400);
    hot_train = hot(:,1:1400);

    validation = subset(1,1401:2000);
    hot_val = hot(:,1401:2000);

    hiddenLayerSize = 16;
    network = patternnet(hiddenLayerSize);

    network.divideParam.trainRatio = 70/100;
    network.divideParam.testRatio = 15/100;
    network.divideParam.valRatio = 15/100;

    % training
    [network, trained] = train(network, subset_vec, hot);
    
elseif ~train_network
    subset=imset_im(2001:4000);
    subset_class = classification(1,2001:4000);

    subset_vec=zeros(784,size(subset,2));
    
    hot = full(ind2vec(double(subset_class),10));
    hot = circshift(hot,[-1 0]);
    
    for a=1:size(subset,2)
        subset_vec(:, a) = reshape(subset{a},1,784);
    end
    
end

% tesing
outputs = network(subset_vec);
errors = gsubtract(hot,outputs);
performance = perform(network, hot, outputs);
plotconfusion(hot,outputs)

%view(network)