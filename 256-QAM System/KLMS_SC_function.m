function [expansionCoefficient,dictionaryIndex,learningCurve,netSizeDiagram] = ...
    KLMS_SC_function(trainInput,trainTarget,reguarization,typeKernel,paramKernel,stepSize)


th1=40;
th2=-4.5;

trainSize=length(trainInput);

expansionCoefficient = stepSize*(trainTarget(1)); 

dictionaryIndex = 1;
dictSize = 1; 


learningCurve = zeros(trainSize,1);
zeroMatrix=zeros(1,length(trainTarget));
learningCurve(1) =mean(abs(trainTarget-zeroMatrix).^2); 

netSizeDiagram = zeros(trainSize,1); 
netSizeDiagram(1) = 1;

for n = 2:trainSize

    predictionVar = reguarization + ker_eval(trainInput(:,n),trainInput(:,n),typeKernel,paramKernel) - max(ker_eval(trainInput(:,n), trainInput(:,dictionaryIndex), typeKernel,paramKernel)./ker_eval(trainInput(:,dictionaryIndex), trainInput(:,dictionaryIndex), typeKernel,paramKernel));
    
    networkOutput = expansionCoefficient*ker_eval(trainInput(:,n),trainInput(:,dictionaryIndex),typeKernel,paramKernel);
    predictionError = abs(trainTarget(n) - networkOutput);
    
    surprise = log(predictionVar)/2 + predictionError^2/(2*predictionVar);
    
    if (surprise < th2 || surprise > th1) 
        learningCurve(n) = learningCurve(n-1); 
        netSizeDiagram(n) = netSizeDiagram(n-1);
        continue;
    end

    dictSize = dictSize + 1;
    dictionaryIndex(dictSize) = n;
    netSizeDiagram(n) = netSizeDiagram(n-1) + 1;

        expansionCoefficient(dictSize) = stepSize*(trainTarget(n)- networkOutput);

        y_out = zeros(trainSize,1);
        for jj = 1:trainSize   
            y_out(jj) = (expansionCoefficient)*...
                ker_eval(trainInput(jj),trainInput(dictionaryIndex),typeKernel,paramKernel);
        end
        err = abs(trainTarget- y_out.');
        learningCurve(n) = mean(err.^2);
end

return

