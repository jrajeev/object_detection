%% Input image and histeq
%I=imread('../calibration_set/Image-3m.png');
I=imread('../color_train/img1.png');
I=imresize(I,[300 400]);
imsz=numel(I(:,:,1));
C = makecform('srgb2lab');
Ilab = applycform(I,C);
Ilab=double(Ilab);
Il=reshape(Ilab(:,:,1),1,imsz);
Ia=reshape(Ilab(:,:,2),1,imsz);
Ib=reshape(Ilab(:,:,3),1,imsz);
Ivec=[Ia;Ib];
d=size(Ivec,1);
%% Use GMM on equalized image
K=3; % #clusters
% Initialize the mean vectors
meanvec=255*rand(d,K);
% RGB space initialization values between 0-1

meanvec(:,1)=[182.3282; 155.7740]; %Barrel Color
meanvec(:,2)=[100; 100]; %Brown door
meanvec(:,3)=[255; 255]; %Brown door
meanvec(:,4)=[120; 50];

% Assign Clusters
prob=zeros(K,size(Ivec,2));
idx=zeros(1,size(Ivec,2));
covmat=diag([20 20]);
%covmat=reshape(covmat,[2 2 1]);
%covmat=repmat(covmat,[1 1 K]);
%covmat(:,:,1)=[50 0 ;0 50];
%covmat(:,:,2)=[50 0 ;0 50];
sigma=15;
for j=1:10
    fprintf ('iteration :: %d\n',j);
    meanvec
    for i=1:K
        %tmp=mvnpdf(Ivec',transpose(meanvec(:,i)),covmat);
        tmp=bsxfun(@minus,Ivec,meanvec(:,i));
        tmp=-1*sum(tmp.*tmp);
        prob(i,:)=exp(tmp/(2*sigma*sigma));
    end
    prob=bsxfun(@rdivide,prob,sum(prob));
   % pidx=idx;
   % [val idx]=max(prob,[],1);
    % Compute new mean vectors
    for i=1:K
        tmp=bsxfun(@times,prob(i,:),Ivec);
        meanvec(:,i)=sum(tmp,2)/sum(prob(i,:),2);
    end
    %if (sum(find(idx-pidx))<500)
    %    fprintf('Kmeans convergence achieved %d\n',j);
    %    break;
    %end
end
fprintf('hello iam %d\n',j);
%% Color segment based on mean vector
for i=1:K
     %tmp=mvnpdf(Ivec',transpose(meanvec(:,i)),covmat);
     %prob(i,:)=tmp';
     tmp=bsxfun(@minus,Ivec,meanvec(:,i));
     tmp=sum(tmp.*tmp);
     prob(i,:)=exp(tmp/(-2*400));
end
prob=bsxfun(@rdivide,prob,sum(prob));
[val idx]=max(prob,[],1);
matidx=reshape(idx,size(I,1),size(I,2));
matidx(matidx~=1)=0;
%matidx(matidx~=0)=1;

imshow(I);
figure,imshow(matidx);
%%
% remove all object containing fewer than 30 pixels
bw = bwareaopen(matidx,100);
% fill a gap in the pen's cap
se = strel('disk',2);
se2 = strel('disk',5);
bw = imclose(bw,se);
bw = imopen(bw,se2);
% fill any holes, so that regionprops can be used to estimate
% the area enclosed by each of the boundaries
bw = imfill(bw,'holes');
figure,imshow(bw)

%%
[B,L] = bwboundaries(bw,'noholes');
stats=regionprops(L,'Centroid','Area','BoundingBox','Perimeter');
for k = 1:length(B)
  area = stats(k).Area;
  % compute the roundness metric
  perimeter=stats(k).Perimeter;
  metric = 4*pi*area/perimeter^2;
  if (metric<0.85 && metric>0.5)
      figure,imshow(I)
      hold on
      boundary = B{k};
      plot(boundary(:,2), boundary(:,1), 'g', 'LineWidth', 2)
      hold on
  end
end
