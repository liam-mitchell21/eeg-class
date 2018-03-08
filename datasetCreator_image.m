clearvars -except imset classification

%
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos',...
    'nndatasets','DigitDataset');
digitData = imageDatastore(digitDatasetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');
%}
is_imset = 0;

while ~is_imset
    imset_im = cell(1,10000);
    placeholdervector = zeros(28*28,1,'uint8');
    parfor a=1:10000
        imset_im{a} =  imread(digitData.Files{a});
    end
    is_imset = 1;
end

classification = digitData.Labels';

randorder = randperm(size(imset_im,2));

classification = classification(randorder);
imset_im = imset_im(randorder);

for b=9000:9005
imagetesty(imset_im{b})

end

save('DigitData.mat','imset_im','classification','randorder','-append')
