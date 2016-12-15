hold on;

filter = ones( 3 ) / 9;
ori = imread('original.png');
ref = imfilter( ori, filter, 'symmetric' );
imwrite( ref, 'reference.png' );
try
    res = imread('result.png');
catch 
    res = uint8( zeros( size(ori) ) );
end

subplot(1,3,1);
imshow(ori);
title('original');
subplot(1,3,2);
imshow(ref);
title('reference');
subplot(1,3,3);
imshow(res);
title('result');

