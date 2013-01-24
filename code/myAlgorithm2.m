function [x y d]=myAlgorithm2(im,K)
szy=size(im,1);
szx=size(im,2);
% Trained Parameters
meanfx=5.5114; %mean focal length in X
meanfy=8.4864; %mean focal length in Y
%Resizing image for faster processing
I=imresize(im,[300 400]);
%Contrast Adjustment using HSV space
Ihsv=rgb2hsv(I);
mtmp=mean(mean(Ihsv(:,:,3)));
Ihsv(:,:,3)=imadjust(Ihsv(:,:,3),[0 mtmp],[mtmp 1]);
Irgb=hsv2rgb(Ihsv);
%Applying adaptive histogram equalization on the R channel
Irgb(:,:,1)=adapthisteq(Irgb(:,:,1));
imsz=numel(I(:,:,1));
Ir=reshape(Irgb(:,:,1),1,imsz);
Ig=reshape(Irgb(:,:,2),1,imsz);
Ib=reshape(Irgb(:,:,3),1,imsz);
Ivec=[Ir;Ig;Ib];
d=size(Ivec,1); %dimension d
% Use K-Means on processed image
%K=5; % #clusters
% Initialize the mean vectors
meanvec=rand(d,K);
% RGB space initialization values between 0-1
% 3 cluster means alone are initialized through training using roipoly
% Rest of the the cluster means are randomnly initialized
meanvec(:,1)=[1; 0.0337; 0.1880]; %Barrel Color
meanvec(:,2)=[0.6;0.0;0.2]; %Brown Close to barrel color
meanvec(:,3)=[1; 0.357; 0.4]; %Chair Red Color
% Assign Clusters
distmat=zeros(1,size(Ivec,2));
thresh=0.01;
idx=zeros(1,size(Ivec,2));
% In K-means the cluster means for the 3 cluster mentioned above are
% limited within range
for j=1:100
    for i=1:K
        tmp=bsxfun(@minus,Ivec,meanvec(:,i));
        distmat(i,:)=sum(tmp.*tmp,1);
    end
    pidx=idx;
    [val idx]=min(distmat,[],1);
    % Compute new mean vectors
    for i=1:K
        tmp=Ivec(:,idx==i);
        meanvec(:,i)=mean(tmp,2);
    end
    if (meanvec(1,1)<0.9 || meanvec(2,1)>0.2)
        meanvec(:,1)=[0.9054; 0.0337; 0.1880];
    end
    if (meanvec(1,2)>0.7 || meanvec(2,2)>0.3)
        meanvec(:,2)=[0.6; 0.0337; 0.1880];
    end
    if (meanvec(1,3)<0.9 || meanvec(2,3)<0.3 || meanvec(2,3)>0.5 || meanvec(3,3)<0.3 || meanvec(3,3)>0.5)
        meanvec(:,3)=[1; 0.337; 0.4];
    end
    if (sum(find(idx-pidx))<500)
        %fprintf('Kmeans convergence achieved %d\n',j);
        break;
    end
end
% Color segment based on mean vector
for i=1:K
    tmp=bsxfun(@minus,Ivec,meanvec(:,i));
    distmat(i,:)=sum(tmp.*tmp,1);
end
tmp=distmat(1,:);
if (mtmp>0.5)
    tmp(tmp>thresh)=1000;
else
    tmp(tmp>0.05)=1000;
end
[val idx]=min(distmat,[],1);
matidx=reshape(idx,size(I,1),size(I,2));
matidx3=matidx;
matidx3(matidx3~=3)=0; % Cluster #3 corresponds to bright red
matidx3(matidx3==3)=1;
matidx(matidx~=1)=0; % Cluster #1 corresponds to normal red
%imshow(Irgb);
%figure,imshow(matidx);
% Post processing of the obtained black-white image to remove noise
% remove all object containing fewer than 30 pixels
% NORMAL RED MASK
bw = bwareaopen(matidx,10);
% fill a gap in the pen's cap
se = strel('disk',5);
se2 = strel('disk',2);
bw = imclose(bw,se);
bw = imopen(bw,se2);
% fill any holes, so that regionprops can be used to estimate
% the area enclosed by each of the boundaries
bw = bwareaopen(bw,100);
bw = imfill(bw,'holes');
%figure,imshow(bw)

% BRIGHT RED MASK
% remove all object containing fewer than 30 pixels
bw3 = bwareaopen(matidx3,10);
% fill a gap in the pen's cap
se = strel('disk',5);
se2 = strel('disk',5);
bw3 = imclose(bw3,se);
bw3 = imopen(bw3,se2);
% fill any holes, so that regionprops can be used to estimate
% the area enclosed by each of the boundaries
bw3 = bwareaopen(bw3,100);
%bw3 = imfill(bw3,'holes');
%figure,imshow(bw3)
% Estimating Centroid and Depth
[B,L] = bwboundaries(bw,'noholes');
stats=regionprops(L,'Centroid','Area','BoundingBox','Perimeter','Orientation');
prevarea=-1;
prevratio=0;
area=-2;

for k = 1:length(B)
  area = stats(k).Area;
  % compute the roundness metric
  centro=stats(k).Centroid;
  %perimeter=stats(k).Perimeter;
  %metric = 4*pi*area/perimeter^2;
  rect1 = stats(k).BoundingBox;
  ratiowh=max(rect1(3),rect1(4))/min(rect1(3),rect1(4));
  arearect=rect1(3)*rect1(4);
  arearatio=area/arearect;
  %[ratio1 ratiowh area]
  if (arearatio>0.5 && ratiowh>1 && ratiowh<1.9)
      if (arearatio>prevratio)
          prevratio=arearatio;
          prevarea=area;
          rectfin=rect1;
          centrofin=centro;
      elseif (arearatio==prevratio && area>prevarea)
          prevarea=area;
          rectfin=rect1;
          centrofin=centro;
          %orientfin=stats(k).Orientation;
      end
  end
end
if (prevarea>0) 
    %figure,imshow(I);
    %hold on
    %rectangle('Position',rectfin,'EdgeColor','g','LineWidth',2);
    %hold on
    %plot(centrofin(1), centrofin(2), 'oy','MarkerFaceColor','y', 'MarkerSize', 2);
    finalcx=centrofin(1);
    finalcy=centrofin(2);
    finalrect=rectfin;
else
    prevarea=-1;
    prevratio=0;
    area=-2;
    [B3,L3] = bwboundaries(bw3,'noholes');
    stats3=regionprops(L3,'Centroid','Area','BoundingBox','Perimeter');
    figure,imshow(I)
    for k = 1:length(B3)
      area = stats3(k).Area;
      % compute the roundness metric
      centro=stats3(k).Centroid;
      %perimeter=stats3(k).Perimeter;
      %metric = 4*pi*area/perimeter^2;
      rect1 = stats3(k).BoundingBox;
      arearect=rect1(3)*rect1(4);
      ratiowh=max(rect1(3),rect1(4))/min(rect1(3),rect1(4));
      arearatio=area/arearect;
      %[ratio1 ratiowh area centro]
      if (arearatio>0.5 && ratiowh>1 && ratiowh<1.9)
          if (arearatio>prevratio)
              prevratio=arearatio;
              prevarea=area;
              rectfin=rect1;
              centrofin=centro;
          elseif (arearatio==prevratio && area>prevarea)
              ratiofin=ratiowh;
              prevarea=area;
              rectfin=rect1;
              centrofin=centro;
          end
      end
    end
    if (prevarea>0) 
        %figure,imshow(I);
        %hold on
        %rectangle('Position',rectfin,'EdgeColor','g','LineWidth',2);
        %hold on
        %plot(centrofin(1), centrofin(2), 'oy','MarkerFaceColor','y', 'MarkerSize', 2);
        finalcx=centrofin(1);
        finalcy=centrofin(2);
        finalrect=rectfin;
    end
end
d1=16*meanfx/finalrect(3);
d2=22*meanfy/finalrect(4);
d=(d1+d2)/2;
x=finalcx*(szx/400);
y=finalcy*(szy/300);
x=round(x);
y=round(y);
if (x<0)
    x=1;
elseif (x>szx)
    x=szx;
end
if (y<0)
    y=1;
elseif (y>szy)
    y=szy;
end
end