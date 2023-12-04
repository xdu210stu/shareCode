function [coeffesRLS,learningCurve] = RLS_Function(train_input_ML,train_target,dimen)
        N=length(train_target);
        lamda=1;
        Pn = eye(dimen)/0.005;
        coeffesRLS = [1;zeros(dimen-1,1)]                                                                                                                                                                                         ;
        for ii = 1:N
            e(ii) = train_target(ii) - train_input_ML(ii,:) * coeffesRLS;
            Kn = Pn * train_input_ML(ii,:)'/( lamda + train_input_ML(ii,:) * Pn * train_input_ML(ii,:)');
            Pn =  (Pn - Kn * train_input_ML(ii,:) * Pn)/ lamda;
            coeffesRLS = coeffesRLS + Kn * e(ii);

        y_out = zeros(N,1);
        for jj = 1:N
            y_out(jj) = train_input_ML(jj,:)*coeffesRLS;
        end
        err = abs(train_target- y_out);
        learningCurve(ii) = mean(err.^2);
        end
end

