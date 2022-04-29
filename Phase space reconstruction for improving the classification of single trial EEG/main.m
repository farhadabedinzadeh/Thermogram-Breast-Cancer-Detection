clc
clear all;
close all;

%% '================ Written by Farhad AbedinZadeh ================'
%                                                                   %

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%filter design for ALPHA band extraction (8-13HZ)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Fs = 128;  % Sampling Frequency

Fstop1 = 7;    % First Stopband Frequency
Fpass1 = 8;    % First Passband Frequency
Fpass2 = 12;   % Second Passband Frequency
Fstop2 = 13;   % Second Stopband Frequency
Dstop1 = 0.1;  % First Stopband Attenuation
Dpass  = 0.1;  % Passband Ripple
Dstop2 = 0.1;  % Second Stopband Attenuation
dens   = 20;   % Density Factor

% Calculate the order from the parameters using FIRPMORD.
[N, Fo, Ao, W] = firpmord([Fstop1 Fpass1 Fpass2 Fstop2]/(Fs/2), [0 1 ...
                          0], [Dstop1 Dpass Dstop2]);

% Calculate the coefficients using the FIRPM function.
b  = firpm(N, Fo, Ao, W, {dens});
Hd1= dfilt.dffir(b);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%filter design for BETA band extraction (13-25HZ)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Fs = 128;  % Sampling Frequency

Fstop1 = 12;    % First Stopband Frequency
Fpass1 = 13;    % First Passband Frequency
Fpass2 = 24;   % Second Passband Frequency
Fstop2 = 25;   % Second Stopband Frequency
Dstop1 = 0.1;  % First Stopband Attenuation
Dpass  = 0.1;  % Passband Ripple
Dstop2 = 0.1;  % Second Stopband Attenuation
dens   = 20;   % Density Factor

% Calculate the order from the parameters using FIRPMORD.
[N, Fo, Ao, W] = firpmord([Fstop1 Fpass1 Fpass2 Fstop2]/(Fs/2), [0 1 ...
                          0], [Dstop1 Dpass Dstop2]);

% Calculate the coefficients using the FIRPM function.
b  = firpm(N, Fo, Ao, W, {dens});
Hd2= dfilt.dffir(b);



%% Load dataset

load 'dataset_BCIcomp1'; % x_train: 1152(9second*128Hz)*3EEG Channel(C3,Cz,C4)*140(Task Repetition)
                         % Y_train 140*1 : (label 1 refering to left-side imagination
                         % and label "2" refers to Right-side imagination
%% Seperate Channels

for i=1:140
   data(:,:)=x_train(:,:,i);
   c3=data(:,1); %C3 
   cz=data(:,2); %Cz 
   c4=data(:,3); %C4 
    
   %% phase space reconstruction

   %%====[Y,T]=phasespace(x,dim,tau)====%% dim(FNN)=2 , tau(TimeDelay)=3
    
   [Y1,T1]=phasespace(c3,2,3);
   [Y2,T2]=phasespace(c4,2,3);
   
   %PhaseSpace Reconstruction Channel C3
   r1=Y1(:,1);
   r2=Y1(:,2);
   %PhaseSpace Reconstruction Channel C4
   r3=Y2(:,1);
   r4=Y2(:,2);

   %Alpha Band
   s1=filter(Hd1,r1);
   s2=filter(Hd1,r2);
   s3=filter(Hd1,r3);
   s4=filter(Hd1,r4);
   %Beta Band
   s5=filter(Hd2,r1);
   s6=filter(Hd2,r2);
   s7=filter(Hd2,r3);
   s8=filter(Hd2,r4);

   %% Feature Extraction

   f1=max(abs(fft(s1)));   %%fft=Fast Fourier transform%%
   f2=max(abs(fft(s2)));
   f3=max(abs(fft(s3)));
   f4=max(abs(fft(s4)));
   f5=max(abs(fft(s5)));
   f6=max(abs(fft(s6)));
   f7=max(abs(fft(s7)));
   f8=max(abs(fft(s8)));

   feature(i,:)=[f1 f2 f3 f4 f5 f6 f7 f8];
   
end

%% create input,output for network

input=feature;
output=y_train-1;

%% classification using "nprtool"

x = input';
t = output';

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
performance = perform(net,t,y);
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




                         
