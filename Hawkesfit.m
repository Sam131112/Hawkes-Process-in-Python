%------------------------------------------------------------------------
% Adaped by Sam from Trey's Code
%
%------------------------------------------------------------------------

close all, clear all, clc;
warning off;
tic %used for timing the algorith

%% Declare User Input
numTrades = 150; % sets the window length
% set pathmame to save images to

file = '/home/sam/Downloads/Hawkes-Process-master/data1.csv';
all_trades = csvread(file);
times = all_trades(:,1); % should be set to the column containing unix timestamps



%% Begin Iteration data points
beginTimes = 1;
endTimes = numTrades;
for iterationTrack = 1:10
%% Initialize current loop varaibles
set = 1:numTrades;
timesNow = times(beginTimes:endTimes);


timesNow = sort(timesNow);

%% Call the minimization function
% set initial guess and reference function
mu0 = .15; % making mu >= or beta seems to work the best
alpha0 = .2; % keeping this about 1/10 of beta seems the best
beta0 = 2;
parameters = [mu0; alpha0; beta0];
func = @(parameters) HawkesMLE(parameters,timesNow);

[fitParameters,logLikelihood,EXITFLAG] = fminunc(func,parameters,optimset('MaxFunEvals',100000,'TolFun',1e-8,'TolX',1e-8));
disp(fitParameters)
end
algoTime = toc %algoTime is the time in seconds for the entire script


