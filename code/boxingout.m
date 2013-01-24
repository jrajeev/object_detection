%%
meanfx=5.5114;
meanfy=8.4864;
%% Input image and histeq
%I=imread('../calibration_set/Image-10m.png');
I=imread('../color_train/img91.png');
I=imresize(I,[300 400]);
%I=double(I);
%I=I/255;
%[Irgb Ihsv]=histo_equalize(I);
Ihsv=rgb2hsv(I);
mtmp=mean(mean(Ihsv(:,:,3)))
Ihsv(:,:,3)=imadjust(Ihsv(:,:,3),[0 mtmp],[mtmp 1]);
%Ihsv(:,:,3)=adapthisteq(Ihsv(:,:,3));
Irgb=hsv2rgb(Ihsv);
Irgb(:,:,1)=adapthisteq(Irgb(:,:,1));
imsz=numel(I(:,:,1));
%Ih=reshape(Ihsv(:,:,1),1,imsz);
%Is=reshape(Ihsv(:,:,2),1,imsz);
%Iv=reshape(Ihsv(:,:,3),1,imsz);
%Ihs=[Ih;Is];
%Ivec=[Ih;Is;Iv];
Ir=reshape(Irgb(:,:,1),1,imsz);
Ig=reshape(Irgb(:,:,2),1,imsz);
Ib=reshape(Irgb(:,:,3),1,imsz);
Ivec=[Ir;Ig;Ib];
%Ivec=double(Ivec);
d=size(Ivec,1);
%% Use K-Means on equalized image
K=5; % #clusters
% Initialize the mean vectors
meanvec=rand(d,K);
% RGB space initialization values between 0-1
meanvec(:,1)=[1; 0.0337; 0.1880]; %Barrel Color
%meanvec(:,1)= [250.5631;   1;   40.3059]; %Barrel Color
meanvec(:,2)=[0.6;0.0;0.2]; %Black
meanvec(:,3)=[1;    0.357;    0.4]; %Blue
%meanvec(:,4)=[0.9869; 0.9704; 0.912]; %Chair Red
%meanvec(:,5)=[ 0.770;0.1354;0.0510]; %Brown
% Assign Clusters
distmat=zeros(1,size(Ivec,2));
thresh=0.01;
idx=zeros(1,size(Ivec,2));
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
        fprintf('Kmeans convergence achieved %d\n',j);
        break;
    end
end
%% Color segment based on mean vector
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
matidx3(matidx3~=3)=0;
matidx3(matidx3==3)=1;
matidx(matidx~=1)=0;
imshow(Irgb);
figure,imshow(matidx);
%%
% remove all object containing fewer than 30 pixels
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
figure,imshow(bw)


%%
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
bw3 = imfill(bw3,'holes');
figure,imshow(bw3)
%%
[B,L] = bwboundaries(bw,'noholes');
stats=regionprops(L,'Centroid','Area','BoundingBox','Perimeter','Orientation');
prevarea=-1;
area=-2;
figure,imshow(bw);
for k = 1:length(B)
  area = stats(k).Area;
  % compute the roundness metric
  centro=stats(k).Centroid;
  perimeter=stats(k).Perimeter;
  metric = 4*pi*area/perimeter^2;
  rect1 = stats(k).BoundingBox;
  orient1=stats(k).Orientation;
  hold on
  mask=drawrect(centro(1),centro(2),rect1(3),rect1(4),orient1);
      maskx=mask(1,:);
    masky=mask(2,:);
    maskx(maskx<=0)=1;
    masky(masky<=0)=1;
    maskx(maskx>size(I,2))=size(I,2);
    masky(masky>size(I,1))=size(I,1);
    ind=sub2ind(size(I),masky,maskx);
    immask=zeros(size(I));
    immask(ind)=1;
    area=sum(sum(immask.*bw));
  ratiowh=max(rect1(3),rect1(4))/min(rect1(3),rect1(4));
  arearect=rect1(3)*rect1(4);
  ratio1=area/arearect;
  [ratio1 ratiowh area]
  if (ratio1>0.7 && ratiowh>1.2 && ratiowh<1.9)
      if (area>prevarea)
          prevarea=area;
          rectfin=rect1;
          centrofin=centro;
          orientfin=orient1;
      end
  end
end
if (prevarea>0) 
    figure,imshow(I);
    hold on
    rectangle('Position',rectfin,'EdgeColor','g','LineWidth',2);
    hold on
    plot(centrofin(1), centrofin(2), 'oy','MarkerFaceColor','y', 'MarkerSize', 2);
else
    prevarea=-1;
area=-2;

[B3,L3] = bwboundaries(bw3,'noholes');
stats3=regionprops(L3,'Centroid','Area','BoundingBox','Perimeter');
figure,imshow(I)
for k = 1:length(B3)
  area = stats3(k).Area;
  % compute the roundness metric
  centro=stats3(k).Centroid;
  perimeter=stats3(k).Perimeter;
  metric = 4*pi*area/perimeter^2;
  rect1 = stats3(k).BoundingBox;
  arearect=rect1(3)*rect1(4);
  ratiowh=max(rect1(3),rect1(4))/min(rect1(3),rect1(4));
  ratio1=area/arearect;
  [ratio1 ratiowh area centro]
  if (ratio1>0.75 && ratiowh>1 && ratiowh<1.9)
      if (area>prevarea)
          ratiofin=ratiowh;
          prevarea=area;
          rectfin=rect1;
          centrofin=centro;
      end
  end
end
if (prevarea>0) 
    figure,imshow(I);
    hold on
    rectangle('Position',rectfin,'EdgeColor','g','LineWidth',2);
    hold on
    plot(centrofin(1), centrofin(2), 'oy','MarkerFaceColor','y', 'MarkerSize', 2);
end
end