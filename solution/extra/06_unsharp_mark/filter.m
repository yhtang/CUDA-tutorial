gauss=fspecial('gaussian',5,1);
lap=[-1,-1,-1;-1,9,-1;-1,-1,-1]/2;
sharp=conv2( gauss, lap, 'full' );
[x,y]=meshgrid([0:1/6:1]);
surf(x,y,sharp);