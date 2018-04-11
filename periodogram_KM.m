%% loading data
clear

load('3d_sensor_epochs.mat')

avg_std=zeros(750,30);
avg_trg=zeros(750,29);

for a=1:30
    temp = epoch_std(:,6,(((a-1)*5)+1):((a*5)));
    avg_std(:,a) = sum(temp, 3)./5;
end
for b=1:29
    temp2 = epoch_trg(:,6,(((b-1)*5)+1):((b*5)));
    avg_trg(:,b) = sum(temp2, 3)./5;
end

data = [avg_std(:,1:25), avg_trg(:,1:24)];

[pxx,w] = periodogram(data);
[M,I] = max(pxx);
data = [M;I]';


test = [avg_std(:,26:30), avg_trg(:,25:29)];

%% fitting model
sumX = 0;
sumY = 0;
for i=1:30
    sumX = sumX + I(i);
    sumY = sumY + M(i);
end
centroid_std_x = sumX/30;
centroid_std_y = sumY/30;

sumX = 0;
sumY = 0;
for i=31:49
    sumX = sumX + I(i)
    sumY = sumY + M(i)
end
centroid_trg_x = sumX/30;
centroid_trg_y = sumY/30;

%dst_between_centroids = sqrt((max(centroid_std_x, centroid_trg_x) - min(centroid_std_x, centroid_trg_x))^2 + ...
    %(max(centroid_std_y, centroid_trg_y) - min(centroid_std_y, centroid_trg_y))^2)

%dst_between_centroids = sqrt((centroid_std_x - centroid_trg_x)^2 + (centroid_std_y - centroid_trg_y)^2);

%% generating predictions

% Not working
function percent = std_percent(x, y)
    dst_to_std = sqrt((centroid_std_x - x)^2 + (centroid_std_y - y)^2);
    dst_to_trg = sqrt((centroid_trg_x - x)^2 + (centroid_trg_y - y)^2);
    percent = dst_to_std/(dst_to_std + dst_to_trg);
end

