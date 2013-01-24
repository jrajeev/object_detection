%% Cross Validation - Train Script
dirstruct = dir('../color_train/*.png');
dd=zeros(5,5);
for K=3:7
    cnt=1;
    for i=1:5%length(dirstruct)
        %im=imread(strcat('../calibration_set/',dirstruct(2).name));
        %im=imread('../color_train/img85.png');
        im=imread(strcat('../color_train/',dirstruct(i).name));
        Imask=imread(strcat('../color_train/mask',dirstruct(i).name));
        [B,L] = bwboundaries(Imask,'noholes');
        stats=regionprops(L,'Centroid');
        centroid=stats(1).Centroid;
        centroid=4*centroid;
        tic
        [x, y, d]=myAlgorithm2(im,K);
        toc
        dd(cnt,:)=sqrt((x-centroid(1))^2 + (y-centroid(2))^2);
        figure,image(im);
        hold on;
        plot(x,y,'g+');
        title(sprintf('Barrel distance: %f m',d));
        hold off; 
    end
    cnt=cnt+1;
end
%%
for i=1:1%size(dd,2)
plot([3:7],ddn(:,i),'LineWidth',2);
hold on
end
hold off