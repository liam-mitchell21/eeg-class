%% loading data


%% fitting model
model = fitcknn(data, classes);

%% generating predictions
label = predict(model,novel_data);