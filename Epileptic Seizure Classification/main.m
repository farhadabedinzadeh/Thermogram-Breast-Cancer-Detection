clc
clear all;
close all;

%% '================ Written by Farhad AbedinZadeh ================'
%                                                                  %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %  
%% load Normal data SET B=> closed eye-non seizure

path='../dataset/B/*.txt' ;  
files=dir(path);

%% Filter Design Using "fdatool" 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%filter design for EEG signal denoising lowpass (Fc=70HZ)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Fs = 173.6;     % Sampling Frequency

Fpass = 69;     % Passband Frequency
Fstop = 70;     % Stopband Frequency
Dpass = 0.1;    % Passband Ripple
Dstop = 0.001;  % Stopband Attenuation
dens  = 20;     % Density Factor

% Calculate the order from the parameters using FIRPMORD.
[N, Fo, Ao, W] = firpmord([Fpass, Fstop]/(Fs/2), [1 0], [Dpass, Dstop]);

% Calculate the coefficients using the FIRPM function.
b  = firpm(N, Fo, Ao, W, {dens});
Hd = dfilt.dffir(b);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Filter D1esign for  ALPHA Band Extraction (8-13HZ)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Fs = 173.6;  % Sampling Frequency

Fstop1 = 7;      % First Stopband Frequency
Fpass1 = 8;      % First Passband Frequency
Fpass2 = 12;     % Second Passband Frequency
Fstop2 = 13;     % Second Stopband Frequency
Dstop1 = 0.001;  % First Stopband Attenuation
Dpass  = 0.1;    % Passband Ripple
Dstop2 = 0.001;  % Second Stopband Attenuation
dens   = 20;     % Density Factor

% Calculate the order from the parameters using FIRPMORD.
[N, Fo, Ao, W] = firpmord([Fstop1 Fpass1 Fpass2 Fstop2]/(Fs/2), [0 1 ...
                          0], [Dstop1 Dpass Dstop2]);

% Calculate the coefficients using the FIRPM function.
b  = firpm(N, Fo, Ao, W, {dens});
Hd1 = dfilt.dffir(b);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%        Filter D1esign for  Beta Band Extraction (13-25HZ)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Fs = 173.6;  % Sampling Frequency

Fstop1 = 12;      % First Stopband Frequency
Fpass1 = 13;      % First Passband Frequency
Fpass2 = 24;     % Second Passband Frequency
Fstop2 = 25;     % Second Stopband Frequency
Dstop1 = 0.001;  % First Stopband Attenuation
Dpass  = 0.1;    % Passband Ripple
Dstop2 = 0.001;  % Second Stopband Attenuation
dens   = 20;     % Density Factor

% Calculate the order from the parameters using FIRPMORD.
[N, Fo, Ao, W] = firpmord([Fstop1 Fpass1 Fpass2 Fstop2]/(Fs/2), [0 1 ...
                          0], [Dstop1 Dpass Dstop2]);

% Calculate the coefficients using the FIRPM function.
b  = firpm(N, Fo, Ao, W, {dens});
Hd2 = dfilt.dffir(b);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%        Filter D1esign for GAMA Band Extraction (13-25HZ)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Fs = 173.6;  % Sampling Frequency

Fstop1 = 7;      % First Stopband Frequency
Fpass1 = 8;      % First Passband Frequency
Fpass2 = 12;     % Second Passband Frequency
Fstop2 = 13;     % Second Stopband Frequency
Dstop1 = 0.001;  % First Stopband Attenuation
Dpass  = 0.1;    % Passband Ripple
Dstop2 = 0.001;  % Second Stopband Attenuation
dens   = 20;     % Density Factor

% Calculate the order from the parameters using FIRPMORD.
[N, Fo, Ao, W] = firpmord([Fstop1 Fpass1 Fpass2 Fstop2]/(Fs/2), [0 1 ...
                          0], [Dstop1 Dpass Dstop2]);

% Calculate the coefficients using the FIRPM function.
b  = firpm(N, Fo, Ao, W, {dens});
Hd1 = dfilt.dffir(b);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%        Filter D1esign for  Beta Band Extraction (26-70HZ)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Fs = 173.6;  % Sampling Frequency

Fstop1 = 26;      % First Stopband Frequency
Fpass1 = 27;      % First Passband Frequency
Fpass2 = 69;     % Second Passband Frequency
Fstop2 = 70;     % Second Stopband Frequency
Dstop1 = 0.001;  % First Stopband Attenuation
Dpass  = 0.1;    % Passband Ripple
Dstop2 = 0.001;  % Second Stopband Attenuation
dens   = 20;     % Density Factor

% Calculate the order from the parameters using FIRPMORD.
[N, Fo, Ao, W] = firpmord([Fstop1 Fpass1 Fpass2 Fstop2]/(Fs/2), [0 1 ...
                          0], [Dstop1 Dpass Dstop2]);

% Calculate the coefficients using the FIRPM function.
b  = firpm(N, Fo, Ao, W, {dens});
Hd3 = dfilt.dffir(b);

%% Closed eye-non seizure Feature etraction


for i = 1:length(files)
    fn = [path(1:end-5) files(i,1).name];
    x=load(fn);
    
    xd=filter(Hd,x);%apply filter for EEG signal denoising lowpass (Fc=70HZ)
    Alpha=filter(Hd1,xd);
    Beta=filter(Hd2,xd);
    Gama=filter(Hd3,xd);
    
    %%%%%%%%%%%Feature Extraction %%%%%%%%%
%% mean PSD   
f1=FMN(Alpha);
f2=FMN(Beta);
f3=FMN(Gama);

%% median PSD
f4=FMD(Alpha);
f5=FMD(Beta);
f6=FMD(Gama);

%% Frequency Ratio
f7=FR(Alpha);
f8=FR(Beta);
f9=FR(Gama);

%% waveform length
f10=WL(Alpha);
f11=WL(Beta);
f12=WL(Gama);

feature_normal(i,:)=[f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12];
   
end
  

%% For Abnormal Data SET C =>  Seizure

path='../dataset/E/*.txt' ;  
files=dir(path);

for i = 1:length(files)
    fn = [path(1:end-5) files(i,1).name];
    x=load(fn);
    
    xd=filter(Hd,x);%apply filter for EEG signal denoising lowpass (Fc=70HZ)
    Alpha=filter(Hd1,xd);
    Beta=filter(Hd2,xd);
    Gama=filter(Hd3,xd);
    
    %%%%%%%%%%%Feature Extraction %%%%%%
%% mean PSD
f1=FMN(Alpha);
f2=FMN(Beta);
f3=FMN(Gama);

%% median PSD
f4=FMD(Alpha);
f5=FMD(Beta);
f6=FMD(Gama);

%% Frequency Ratio
f7=FR(Alpha);
f8=FR(Beta);
f9=FR(Gama);

%% waveform length
f10=WL(Alpha);
f11=WL(Beta);
f12=WL(Gama);

feature_seizure(i,:)=[f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12];
   
end


%% Plot The Sample

subplot(4,1,1)
t=linspace(0,23.6,4097);
plot(t,xd)
title('EEG Signal Denoising Lowpass')

subplot(4,1,2)
plot(t,Alpha)
title('EEG Alpha Band Signal')

subplot(4,1,3)
plot(t,Beta)
title('EEG Beta Band Signal')


subplot(4,1,4)
plot(t,Gama)
title('EEG Gama Band Signal')


%% Creating Network

Input=[feature_normal,feature_seizure];
Output=[zeros(length(feature_normal),1),ones(length(feature_seizure),1)];


%% classification using "nprtool"

x = Input';
t = Output';

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network
hiddenLayerSize = 4;
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



