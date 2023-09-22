clear;
load('data_v1_s3.mat');
% z x y
vel3d = reshape(rec_data,[1001,20,20]);
vel3d = permute(vel3d,[3 2 1]);
x = 1:1:20;
y = 1:1:20;
z = 1:1:1001;
[X,Y,Z] = meshgrid(x,y,sort(z,'descend'));

V = vel3d;

xslice = [1,20];   
yslice = [1,20];
zslice = [1,1001];
% V(1:100,1:300,1:70) = NaN;
h = slice(X,Y,Z,V,xslice,yslice,zslice);
set(h,'FaceColor','interp','EdgeColor','none')
camproj perspective
axis tight
grid off
box on
colormap gray
colorbar;
caxis([-0.5 0.5]);
xlabel('y');ylabel('x');zlabel('Time ms');
% set(gca,'ZTick',0.001:0.0001:1);