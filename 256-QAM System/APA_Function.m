function [coeffesAPA,learningCurve] = APA_Function(train_input_ML,train_target,dimen,K,stepSize,paramRegularization)
        N=length(train_target);
        coeffesAPA = zeros(dimen,1); 
        train_input_ML=train_input_ML.';
        for ii=1:N
            Output = train_input_ML(:,ii).'*conj(coeffesAPA);
            if ii<K
                coeffesAPA= coeffesAPA+stepSize*conj(train_target(ii)-Output)*train_input_ML(:,ii);
            else
             X=train_input_ML(:,ii-K+1:ii);
             Y=train_target(ii-K+1:ii);
             ei=Y.'-X.'*conj(coeffesAPA);
            coeffesAPA= coeffesAPA+stepSize*X*pinv(X.'*X+paramRegularization*eye(K))*conj(ei);
            end
        y_out = zeros(N,1);
        for jj = 1:N
            y_out(jj) = train_input_ML(:,jj).'*conj(coeffesAPA);
        end
        err = abs(train_target.'- y_out);
        learningCurve(ii) = mean(err.^2);
        end
end

