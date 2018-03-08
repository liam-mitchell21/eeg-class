clearvars -except imset classification

%
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos',...
    'nndatasets','DigitDataset');
digitData = imageDatastore(digitDatasetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');
%}
is_imset = 0;

while ~is_imset
    imset = zeros(784,10000,'uint8');
    placeholdervector = zeros(28*28,1,'uint8');
    parfor a=1:10000
        imdat = imread(digitData.Files{a});
        imset(:,a) = reshape(imdat,28*28,1);
    end
    is_imset = 1;
end

classification = digitData.Labels';

randorder = randperm(size(imset,2));

classification = classification(randorder);
imset = imset(:,randorder);

for b=9000:9005
imagetesty(imset(:,b))
end

save('DigitData.mat','imset','classification','randorder')
