function mask=drawrect(x,y,xlen,ylen,angle)
    angle=angle*pi/180;
    rotmat=[cos(angle) -sin(angle); sin(angle) cos(angle)];
    xcoor=[-xlen/2 xlen/2 xlen/2 -xlen/2];
    coorx=[round(-xlen/2):1:round(xlen/2)];
    coory=[round(-ylen/2):1:round(ylen/2)];
    coord=[coorx;coory];
    rotcoord=rotmat*coord;
    rotcoord=rotcoord+[x;y];
    mask=int(rotcoord);
    ycoor=[-ylen/2 -ylen/2 ylen/2 ylen/2];
    coor=[xcoor;ycoor];
    rotcoor=rotmat*coor;
    rotx=rotcoor(1,:);
    roty=rotcoor(2,:);
    rotx = ceil(rotx+x);
    roty =ceil(roty+y);
    rotx =[rotx rotx(1)];
    roty =[roty roty(1)];
    plot(rotx,roty,'EdgeColor','g','LineWidth',2);
end