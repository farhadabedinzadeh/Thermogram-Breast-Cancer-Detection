clc
clear all;
close all;

%% '================ Written by Farhad AbedinZadeh ================'
%                  Special Thanks to Dr.NajafAbadian              %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
%% Load Data

load 'dataset_BCIcomp1.mat';

%dar in qesmat channelHa ra az yekDigar joda mikonim
for i=1:140
   data(:,:)=x_train(:,:,i); %yek matrix 2D mikhahim 
   c3=data(:,1); %C3 column1 az data hast
   cz=data(:,2); %Cz column2 az data hast
   c4=data(:,3); %C4 column3 az data hast
    
   %hal PhaseSpace Reconstruction ra baraye channelHaye C3 o C4 anjam midahim va BandHaye Alpha o Beta
   %ra estekhraj mikonim.%%az "phasespace" farakhani mikonim.
   %%====[Y,T]=phasespace(x,dim,tau)====%% dim(FNN)=2 , tau(TimeDelay)=3
    
   [Y1,T1]=phasespace(c3,2,3);
   [Y2,T2]=phasespace(c4,2,3);

    %hal Time Sries ijad mikonim.
   
   %PhaseSpace Reconstruction Channel C3
   r1=Y1(:,1);
   r2=Y1(:,2);
   %PhaseSpace Reconstruction Channel C4
   r3=Y2(:,1);
   r4=Y2(:,2);
   
   x=[r1;r2;r3;r4];
   
   %% Now Using Wavelet
      
         %%Denoise%%
       
    wname='db10'; %WaveName
    [C,L]=wavedec(x,4,wname);   %[Coefficient,Length]=WaveDecomposition(signalDOKHTAR,level tajzie,WaveName=dabichiz10)
                                 %Coefficient Zarayebe Koliat(Approximation) va Jozeiat(Detail) hast.
[THR,SORH,KeepApp]=ddencmp('den','wv',x);  %[Threshold,SoftOrHard,KeepApproximation]=DefaultDenoisingComposition(Chikar?='Denoising',BaChi?='Wavelet',Kio?=SignalDokhtar);
x= wdencmp('gbl',C,L,wname,4,THR,SORH,KeepApp);  %%SignalDokhtar=x=WaveletDenoisingCompression('gosi bell shekl',C,L,'WaveName',THR,SORH,KeepApp);
    
    
WaveletFunction='db8'; %OR 'sym8' Symlet8
    [C,L]=wavedec(x,8,WaveletFunction);
    %% Calculate The Coefficient Vector
    
    %%%CoefficientDtail=DetailCoefficient(C,L,level);
    
    cD1=detcoef(C,L,1);  %Noisy
    cD2=detcoef(C,L,2);  %Noisy
    cD3=detcoef(C,L,3);  %Noisy
    cD4=detcoef(C,L,4);  %Noisy
    cD5=detcoef(C,L,5);  %Gama
    cD6=detcoef(C,L,6);  %Beta
    cD7=detcoef(C,L,7);  %Alpha
    cD8=detcoef(C,L,8);  %Teta
    cA8=appcoef(C,L,WaveletFunction,8);  %Delta   %CoefficientApproximation=ApproximationCoefficient(C,L,level)
    
    
    %%%%Calculate The Details Vector
    
    %WaveletRecompositionCoefficient=(ChioMikhad?'detail',C,L,WaveletFunction,level)
    
    D1=wrcoef('d',C,L,WaveletFunction,1);    %Noisy
    D2=wrcoef('d',C,L,WaveletFunction,2);    %Noisy
    D3=wrcoef('d',C,L,WaveletFunction,3);    %Noisy
    D4=wrcoef('d',C,L,WaveletFunction,4);    %Noisy
    D5=wrcoef('d',C,L,WaveletFunction,5);    %Gama
    D6=wrcoef('d',C,L,WaveletFunction,6);    %Beta
    D7=wrcoef('d',C,L,WaveletFunction,7);    %Alpha
    D8=wrcoef('d',C,L,WaveletFunction,8);    %Teta
    A8=wrcoef('d',C,L,WaveletFunction,8);    %Delta
   
    
    f1=max(abs(fft(cD6)));  %BETA
    f2=max(abs(fft(cD7)));  %ALPHA
    f3=max(abs(fft(D6)));   %BETA
    f4=max(abs(fft(D7)));   %ALPHA
    f5=FMD(cD6);
    f6=FMD(cD7);
    f7=FMN(cD6);
    f8=FMN(cD7);
    f9=FR(cD6);
    f10=FR(cD7);
    f11=WL(cD6);
    f12=WL(cD7);
    
   
    f13=FMD(D6);
    f14=FMD(D7);
    f15=FMN(D6);
    f16=FMN(D7);
    f17=FR(D6);
    f18=FR(D7);
    f19=WL(D6);
    f20=WL(D7);
    
    [Ea,Ed]=wenergy(C,L);  %%Wavelet Energy
    
     E=wentropy(x,'shannon');  %%Wavelet Entropy 
    
    %%wenergy va wentropy ra be onvane 2 feature jadid estefade kardim ke
    %%bebinim natije che taghiri mikonad.
    
    
    Feature(i,:)=[f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16 f17 f18 f19 f20 Ea Ed E];
end

%% Creating Input & Output

Input=Feature;
Output=y_train-1;

%% Simple Sc Or Advanced Sc By "nnstart"


x = Input';
t = Output';

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network
hiddenLayerSize = 5;
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







