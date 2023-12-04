function [expansionCoefficient,dictionaryIndex,learningCurve,netSizeDiagram] = ...
    KAPA_SC_function(K,trainInput,trainTarget,paramRegularization,typeKernel,paramKernel,stepSize)

th1=40;
th2=-3;

trainSize=length(trainInput);
networkOutput = zeros(K,1); 

expansionCoefficient = stepSize*trainTarget(1); 

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
        expansionCoefficient(dictSize) = stepSize*predictionError;
    else
        for kk = 1:K-1
            networkOutput(K-kk) = expansionCoefficient*ker_eval(trainInput(:,dictionaryIndex(dictSize-kk)),trainInput(:,dictionaryIndex(1:end-1)),typeKernel,paramKernel); 
        end
        aprioriErr = trainTarget(dictionaryIndex(end-K+1:end))' - networkOutput; 
        expansionCoefficient(dictSize) = 0;
        expansionCoefficient(dictSize-K+1:dictSize) = expansionCoefficient(dictSize-K+1:dictSize) + ... 
            stepSize*aprioriErr'*inv(paramRegularization*eye(K) + gramMatrix((trainInput(dictionaryIndex(end-K+1:end))),typeKernel,paramKernel));
    end
        y_te = zeros(trainSize,1);
        for jj = 1:trainSize   
            y_te(jj) = expansionCoefficient*ker_eval(trainInput(jj),trainInput(dictionaryIndex),typeKernel,paramKernel);
        end
        err = trainTarget- y_te';
        learningCurve(n) = mean(err.^2);

end

return

