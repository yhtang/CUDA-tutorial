hold on;

ori = imread('original.png');
ref = imread('reference.png');
try
    res = imread('result.png');
catch 
    res = uint8( zeros( size(ori) ) );
end

subplot(2,3,1);
imshow(ori);
title('original');
subplot(2,3,2);
imshow(ref);
title('reference');
subplot(2,3,3);
imshow(res);
title('result');

rangex = 500:590;
rangey = 600:760;
subplot(2,3,4);
imshow(ori(rangex,rangey));
title('original');
subplot(2,3,5);
imshow(ref(rangex,rangey));
title('reference');
subplot(2,3,6);
imshow(res(rangex,rangey));
title('result');
