
clear all;
close all;
tic

train_length=200;

KAPA_K=5; 
paramRegularization=0.001;
typeKernel='Gauss';
paramKernel=70;
stepSize=0.02;     
stepSize_klms=1;
dimen=13;
      
load("train_input.mat");
load("train_target.mat");
load("train_input_ML.mat")
        
        
 for times=1:train_length 
       [wopt_RLS,learningCurve_rls(times,:)] = RLS_Function(train_input_ML{times},train_target(times,:),dimen);
       [wopt_APA,learningCurve_apa(times,:)] = APA_Function(train_input_ML{times},train_target(times,:),dimen,KAPA_K,stepSize,paramRegularization);
       [expCoeffients_kapa,dictionaryIndex_kapa,learningCurve_kapa(times,:),netSize_kapa(times,:)] = KAPA_SC_function(KAPA_K,train_input(times,:),train_target(times,:),paramRegularization,typeKernel,paramKernel,stepSize);
       [expCoeffients_klms,dictionaryIndex_klms,learningCurve_klms(times,:),netSize_klms(times,:)] = KLMS_SC_function(train_input(times,:),train_target(times,:),paramRegularization,typeKernel,paramKernel,stepSize_klms);
       [expCoeffients_mk,dictionaryIndex_mk,learningCurve_mk(times,:),netSize_mk(times,:)] = MSER_KAPA_SC_function(KAPA_K,train_input(times,:),train_target(times,:),paramRegularization,typeKernel,paramKernel,stepSize);
end
learningCurve_rls=mean(learningCurve_rls,1);
learningCurve_apa=mean(learningCurve_apa,1);
learningCurve_klms=mean(learningCurve_klms,1);
learningCurve_kapa=mean(learningCurve_kapa,1);
learningCurve_mk=mean(learningCurve_mk,1);                            
netSize_kapa=mean(netSize_kapa,1);
netSize_klms=mean(netSize_klms,1);
netSize_mk=mean(netSize_mk,1);

   figure;
   semilogy(learningCurve_rls);
   hold on
   semilogy(learningCurve_apa);
   hold on
   semilogy(learningCurve_klms);
   hold on
   semilogy(learningCurve_kapa); 
    hold on
   semilogy(learningCurve_mk);
   xlabel('Iteration');
   ylabel('MSE');
   legend('Poly-RLS','Poly-APA','KLMS-SC','KAPA-SC','MSER-KAPA-SC');
   
   figure;
   plot(netSize_klms);
   hold on
   plot(netSize_kapa);
   hold on
   plot(netSize_mk);
   xlabel('Iteration');
   ylabel('Dictionary-Size');
   legend('KLMS-SC','KAPA-SC','MSER-KAPA-SC');
