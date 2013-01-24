%% Test Script
dirstruct = dir('../train_set/*.png');
for i=1:length(dirstruct)
    %im=imread(strcat('../calibration_set/',dirstruct(2).name));
    %im=imread('../color_train/img85.png');
    im=imread(strcat('../train_set/',dirstruct(i).name));
    tic
    [x, y, d]=myAlgorithm(im);
    toc
    image(im);
    hold on;
    plot(x,y,'g+');
    title(sprintf('Barrel distance: %f m',d));
    hold off;
    pause; 
end
