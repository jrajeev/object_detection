function [Irgb Ihsv]=histo_equalize(I)
    Ihsv=rgb2hsv(I);
    Ihsv(:,:,3)=histeq(Ihsv(:,:,3));
    Irgb=hsv2rgb(Ihsv);    
end