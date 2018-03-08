
%% data load
load('DigitData.mat','imset_im','classification')

subset = imset_im(1,1:2000);
subset_class = classification(1,1:2000);

hot = full(ind2vec(double(subset_class),10));
hot = circshift(hot,[-1 0]);

training = subset(1,1:1400);
hot_train = hot(:,1:1400);

validation = subset(1,1401:2000);
hot_val = hot(:,1401:2000);


%% layer1

hiddensize1 = 100; 

autoenc1 = trainAutoencoder(training,hiddensize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);
feature1 = encode(autoenc1,training);
%% layer 2

hiddensize2 = 50; 

autoenc2 = trainAutoencoder(feature1,hiddensize2, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);
feature2 = encode(autoenc2,feature1);
%% layer 3

softnet = trainSoftmaxLayer(feature2,hot_train,'MaxEpochs',400);

%% stick em together

deepnet = stack(autoenc1,autoenc2,softnet);

%% validation
test = zeros(784,600);
for a=1:600
    test(:,a) = reshape(validation{1,a},1,784);
end

y = deepnet(test);
conf = plotconfusion(hot_val,y);

savefig(conf,'stack_autoenc_cmat.fig')

