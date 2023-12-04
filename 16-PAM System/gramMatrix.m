function G = gramMatrix(data,typeKernel,paramKernel)

[inputDimension,dataSize] = size(data);
G = zeros(dataSize,dataSize);

for ii = 1:dataSize
    jj = ii:dataSize;
    G(ii,jj) = ker_eval(data(:,ii),data(:,jj),typeKernel,paramKernel);
    G(jj,ii) = G(ii,jj);
end
return