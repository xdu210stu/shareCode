function [expansionCoefficient,dictionaryIndex,learningCurve,netSizeDiagram] = ...
    MSER_KAPA_SC_function(K,trainInput,trainTarget,paramRegularization,typeKernel,paramKernel,stepSize)

trainSize=length(trainInput);
networkOutput = zeros(K,1); 
Amplitude_float=0.2;

th1=40;
th2=-3;

beta=3;

expansionCoefficient = -stepSize*(tanh(beta*(-trainTarget(1)+1))+tanh(beta*(-trainTarget(1)-1)))/ker_eval(trainInput(1),trainInput(1),typeKernel,paramKernel); 

dictionaryIndex = 1;
dictSize = 1;

learningCurve = zeros(trainSize,1);
learningCurve(1) = mean(trainTarget.^2);  


netSizeDiagram = zeros(trainSize,1); 
netSizeDiagram(1) = 1;

for n = 2:trainSize

    if dictSize <= K
        predictionVar = paramRegularization + ker_eval(trainInput(:,n),trainInput(:,n),typeKernel,paramKernel) - max(ker_eval(trainInput(:,n), trainInput(:,dictionaryIndex), typeKernel,paramKernel)./ker_eval(trainInput(:,dictionaryIndex), trainInput(:,dictionaryIndex), typeKernel,paramKernel));
    else
        GG=inv(paramRegularization*eye(K) + gramMatrix((trainInput(dictionaryIndex(end-K+1:end))),typeKernel,paramKernel));
        hu=ker_eval(trainInput(:,n),trainInput(:,dictionaryIndex(end-K+1:end)),typeKernel,paramKernel);
        predictionVar = paramRegularization + ker_eval(trainInput(:,n),trainInput(:,n),typeKernel,paramKernel) - hu.'*GG*hu;
    end
    networkOutput(K) = expansionCoefficient*ker_eval(trainInput(:,n),trainInput(:,dictionaryIndex),typeKernel,paramKernel);
    predictionError = trainTarget(n) - networkOutput(K);
    
    surprise = log(predictionVar)/2 + predictionError^2/(2*predictionVar);
    
    if (surprise < th2 || surprise > th1) 
        learningCurve(n) = learningCurve(n-1); 
        netSizeDiagram(n) = netSizeDiagram(n-1);
        continue;
    end

    dictSize = dictSize + 1;
    dictionaryIndex(dictSize) = n;
    netSizeDiagram(n) = netSizeDiagram(n-1) + 1;
    
    if dictSize < K 
        expansionCoefficient(dictSize) =  -stepSize*(tanh(beta*(networkOutput(K)-trainTarget(n)+Amplitude_float))+tanh(beta*(networkOutput(K)-trainTarget(n)-Amplitude_float)))/ker_eval(trainInput(n),trainInput(n),typeKernel,paramKernel);
    else
        for kk = 1:K-1
            networkOutput(K-kk) = expansionCoefficient*ker_eval(trainInput(:,dictionaryIndex(dictSize-kk)),trainInput(:,dictionaryIndex(1:end-1)),typeKernel,paramKernel); 
        end
        aprioriErr = networkOutput-trainTarget(dictionaryIndex(end-K+1:end))'; 
        Ik=tanh(beta*(aprioriErr+Amplitude_float*ones(K,1)))+tanh(beta*(aprioriErr-Amplitude_float*ones(K,1)));
        expansionCoefficient(dictSize) = 0;
        expansionCoefficient(dictSize-K+1:dictSize) = expansionCoefficient(dictSize-K+1:dictSize) - ... 
            stepSize*Ik'*inv(paramRegularization*eye(K) + gramMatrix((trainInput(dictionaryIndex(end-K+1:end))),typeKernel,paramKernel));
    end    

        y_out = zeros(trainSize,1);
        for jj = 1:trainSize   
            y_out(jj) = expansionCoefficient*ker_eval(trainInput(jj),trainInput(dictionaryIndex),typeKernel,paramKernel);
        end
        err = trainTarget- y_out';
        learningCurve(n) = mean(err.^2);
end

return

