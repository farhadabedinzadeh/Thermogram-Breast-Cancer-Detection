clc
clear all;
close all;

%% '================ Written by Farhad AbedinZadeh ================'

%% step by step for beginners

%% load a signal

load('16265m.mat');

%% If our signal have multiple channel,seperate the chennel

ecg1=val(1,:);
ecg2=val(2,:);

%% If we have the sampling frequency(fs) and want to determine the duration of our signal
 

fs=128 %frequency sampling
time=length(ecg1)/fs

%% If we know length of signal and want to detect sampling frequency


timee=3600 %Second (1hour=3600second)
freqs=length(ecg1)/timee

%% If we want to cut 10 second of signal(*)

ecg10s=ecg1(1,1:1:10*fs);
figure;
plot(ecg10s);grid minor;axis tight,title('Raw Signal')
xlabel('Sample');ylabel('voltage(MicroVolt)');

%% change the horizontal axis into time(sec)

t=linspace(0,10,10*fs);
figure;
plot(t,ecg10s);grid minor;title('Raw Signal')
xlabel('Time(sec)');ylabel('voltage(MicroVolt)');

%% Extract 2nd two seconds

%%(0s===2s===4s)====>> (1-256  samples occurs in fist second) ====>> (257-512 samples occurs in 2nd seconds)  

ecg2s2=ecg1(1,2*fs+1:1:4*fs);
t2=linspace(2,4,2*fs);
figure;
plot(t2,ecg2s2);grid minor;
title('ECG 2s2');
xlabel('time(s)');
ylabel('voltage(MicroVolt)');

%% 4th two seconds - 8th two seconds - 17th two seconds


                         %formula: ((n*2)-2)*fs : (n*2)*fs


Sum_ecg(:,1)= ecg1(1,6*fs+1:1:8*fs)';     %% 4th two seconds
Sum_ecg(:,2)= ecg1(1,14*fs+1:1:16*fs)' ;  %%8th two seconds
Sum_ecg(:,3)= ecg1(1,32*fs+1:1:34*fs)' ;  %%17th two seconds


%% Segment 1 minute to 1 minute's of signal and drop them into separate rows

c=1;     %start 1min
d=60*fs; %end 1min

for i=1:1:60;
  ecg_total(i,:)=ecg1(1,c:d);
  c=c+60*fs;
  d=d+60*fs;
end

%% Add white gaussian noise to 4th seconds of 10 second ecg(which created on line 32)

% ecg10s = ecg1(1,1:1:10*fs);
t = linspace(0,10,10*fs);
ecg10s(1,3*fs+1:4*fs)= ecg10s(1,3*fs+1:4*fs)+100*randn(1,128);
figure;
plot(t,ecg10s);grid minor
title('Noisy ECG With Guassian White Noise');


%% Amplitude Modulation of 0.5Hz Sinusoidal Respiratory noise

ecg10s = ecg1(1,1:1:10*fs);
t = linspace(0,10,10*fs);
f = 0.5 %Hz
resp_noise = 100*sin(2*pi*f*t);  %2πωt
noisy_ecg = ecg10s.* resp_noise ; 
figure;
plot(t,noisy_ecg), grid minor
title('Noisy ECG With Respiratory 0.5Hz');

%% R peaks with simple function using https://github.com/DavidLoon

xECG=ecg10s;
fsECG=fs;
[xRRI,fsRRI]=ECG_to_RRI(xECG,fsECG);
