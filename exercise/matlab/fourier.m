%% load image
img = imread( 'iron-man.jpg' );
img_gray = rgb2gray( img );

%% do FFT on CPU
fprintf('performing FFT on CPU\n');
tic; fft_cpu = fft2( img_gray ); toc
spectral_cpu = log( abs( fftshift( fft_cpu ) ) );

%% do FFT on GPU
fprintf('performing FFT on GPU\n');
try
    img_gpu = gpuArray( img_gray );
    tic; fft_gpu = fft2( img_gpu ); toc
    fft_gpu = gather( fft_gpu );
    spectral_gpu = log( abs( fftshift( fft_gpu ) ) );
end

%% visualize result
close all
figure(1); imshow(img)
figure(2); imagesc( spectral_cpu ); colormap jet
try
    figure(3); imagesc( spectral_gpu ); colormap jet
end