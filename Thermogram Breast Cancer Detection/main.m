clc
clear all;
close all;

%% '================ Written by Farhad AbedinZadeh ================'
%                                                                 %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %                                                                
%% 
path='.\total_dataset\*.jpg' ;
files=dir(path);

for i = 1:length(files)
    fn = [path(1:end-5) files(i,1).name];
    Im = imread(fn);
    Im1 = rgb2gray(Im);
%     figure;
%     imshow(Im1);
    
    Im2 = histeq(Im1);
%     figure;
%     imshow(Im2);
    
    Im3 = medfilt2(Im2, [3,3]);
%     figure;
%     imshow(Im3);
    
    sigma = 0.4;
    alpha = 0.5;
    Im4 = locallapfilt(Im3,sigma,alpha);
    
    %% Feature Extraction 
    Im5 = im2double(Im4);
    f1=mean(Im5(:));
    f2=var(Im5(:));
    f3=std(Im5(:));
    f4=max(Im5(:));
    f5=min(Im5(:));
    f6=entropy(Im5(:));
    f7=kurtosis(Im5(:));
    f8=skewness(Im5(:));

%     Feature_Stat(i,:)=[f1 f2 f3 f4 f5 f6 f7 f8];  %%%Statictical feature,Accuracy is not good.
    
%     Feauture_LBP(i,:)=extractLBPFeatures(Im5);    %%Accuracy 100%
    
      Feature_HOG(i,:)=extractHOGFeatures(Im5);    %%Accuracy 100%
    
end



%% Input & Output

%Input=Feature_Stat; %chon Feature_Stat khub nabud
%Input=Feauture_LBP;  %%Accuracy 100%
Input=Feature_HOG;  %%Accuracy 100%
Output=[ones(100,1);zeros(100,1)];


%% Classification Using nprtool


%   Input - input data.
%   Output - target data.

x = Input';
t = Output';

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network
hiddenLayerSize = 3;
net = patternnet(hiddenLayerSize, trainFcn);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 5/100;

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y)
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);

% View the Network
% view(net)

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
figure, plotconfusion(t,y)
%figure, plotroc(t,y)