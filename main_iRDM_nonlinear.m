%% Implementation of the iRDM algorithm in the following paper:
%%
%% LIU Zi-Ang, JIANG Xue, WU Dong-Rui. Pool-Based Unsupervised Active Learning for Regression Using Iterative
%% Representativeness-Diversity Maximization (iRDM), Pattern Recognition Letters, 2020.
%%
%% Compare 6 approaches on nonlinear (kernel) regression model (RBF-SVR)
%%
%% 1. RS (Random sampling)
%%
%% 2. GSx in the paper: D. Wu, C-T Lin and J. Huang*, "Active Learning for Regression Using Greedy Sampling,"
%%    Information Sciences, vol. 474, pp. 90-105, 2019.
%%
%% 3. RD in the paper: D. Wu, "Pool-based sequential active learning for regression,"
%%    IEEE Trans. on Neural Networks and Learning Systems, 30(5), pp. 1348-1359, 2019.
%%
%% 4. iRDM 
%%
%% 5. QBC in the paper: T. RayChaudhuri, L. G. Hamey, "Minimisation of data collection by active learning,"
%%    in: Proc. IEEE Int¡¯l. Conf. on Neural Networks, Vol. 3, Perth, Australia, 1995, pp. 1338-1341.
%%
%% 6. RSAL in the paper: F. Douak, F. Melgani, N. Benoudjit, "Kernel ridge regression with active learning for wind speed prediction,"
%%    Applied Energy 103 (2013) 328¨C340.
%%
%% Dongrui Wu, drwu@hust.edu.cn

clc; clearvars; close all; rng('default');
datasets = {'Concrete-CS','Tecator-Moisture'};
nDatasets = length(datasets);
nRepeats = 100; % Number of repeats to get statistically significant results
minN = 5;       % mininum number of samples to select
maxN = 50;      % maximum number of samples to select
Cmax = 5;       % Maximum number of iterations for iRDM; c_max in the paper
rr =   0.01;    % L2 regularization coefficient for each model 
lambda=0.01;    % lambda for each kernel model, 0 for linear model
C = 1/(2*rr);         % relation: rr of Ridge and C for SVR
ks = 1/sqrt(lambda);  % relation: ks of matlab function "fitrsvm" and lambda of RBF kernel
nBoots = 4;           % Parameters for QBC
train_test_rate = 0.5;% Train-test rate in paper

% ALR algorithms for RR model (1-4: unsupervised ALR; 5-6: supervised ALR)
Algs_list =   {'RS', 'GSx', 'RD', 'iRDM', 'QBC', 'RSAL'};
Algs_ids =    [ 1,     2,    3,    4,       5,     6   ];
plotAlgs_list =  {'RS', 'QBC', 'RSAL', 'GSx', 'RD', 'iRDM'};
plotAlgsIds =    [ 1,     5,     6,      2,     3,    4   ];
Color =     {'k',   [1,0.7,0.5],  'm',    'b',   'g',  'r'};
LineStyle = {':',    '--',        '--',   '-',   '-',  '-'};
nAlgs = length(Algs_ids);

RMSEs = cell(1,nDatasets);
CCs = RMSEs;
for s = 1:nDatasets
    RMSEs{s} = nan(nAlgs,nRepeats,maxN);CCs{s} = RMSEs{s};

    temp=load([datasets{s} '.mat']); data = temp.data;
    X0=data(:,1:end-1); Y0=data(:,end); nY0=length(Y0);
    
	for r = 1:nRepeats
        [s r]
        
        % random effect: 50% as pool
        ids = datasample(1:nY0,round(nY0 * train_test_rate),'Replace',false);
        idsI = 1:nY0; idsI(ids) = [];
        
        % normalization
        X = X0(ids,:); Y = Y0(ids); nY = length(Y);
        XI = X0(idsI,:); YI = Y0(idsI); nYI = length(YI);
        [X,mu,sigma] = zscore(X);   XI = (XI-mu)./sigma;
        distX=pdist2(X,X);
        
        %% 1. Random selection (RS)
        idsTrain = repmat(sort(datasample(1:nY, maxN,'replace',false)),nAlgs,1);
        
        for d = minN:maxN
            C_ = max(1,ceil( (d-1) * rand(nBoots,d-1) ) ); % For QBC
            
            %% 2: GSx
            if d==minN
                dist = mean(distX,2);  % Initialization for GSx
                [~,idsTrain(2,1)] = min(dist);
                idsRest = 1:nY; idsRest(idsTrain(2,1)) = [];
                for n = 2:minN
                    dist = min(distX(idsRest,idsTrain(2,1:n-1)),[],2);
                    [~,idx] = max(dist);
                    idsTrain(2,n) = idsRest(idx);
                    idsRest(idx) = [];
                end
            else
                idsRest = 1:nY; idsRest(idsTrain(2,1:d-1)) = [];
                dist = min(distX(idsRest,idsTrain(2,1:d-1)),[],2);
                [~,idx] = max(dist);
                idsTrain(2,d) = idsRest(idx);
            end
            
            %% 3: RD
            [idsCluster,~,~,Dist] = kmeans(X,d,'MaxIter',200,'Replicates',5);
            idsP=cell(1,d);
            for n = 1:d  % find the one closest to the centroid
                idsP{n}=find(idsCluster == n);
                [~,idx] = min(Dist(idsP{n},n));
                idsTrain(3,n) = idsP{n}(idx);
            end
            
            %% 4: iRDM
            P = idsTrain(3,1:d);
            R = nan(nY, 1);  % representaiveness
            for n = 1:nY
                R(n) = sum( distX(n, idsP{idsCluster(n)}) )/(length(idsP{idsCluster(n)})-1);
            end
            for c = 1:Cmax
                for n = 1:d
                    idsFix = P(end,:); idsFix(n) = [];
                    D = min( distX(idsP{n}, idsFix), [], 2 );   % diversity
                    [~,idx] = max(D-R(idsP{n}));
                    idsTrain(4,n) = idsP{n}(idx);
                end
                if ismember(idsTrain(4,1:d), P, 'rows')
                    break;
                else
                    P(c+1,:) = idsTrain(4,1:d);
                end
            end
            
            %% 5: QBC; the first minN samples were obtained randomly
            if d == minN
                idsTrain(5,1:d)=idsTrain(1,1:d);
            else
                Ys=repmat(Y,1,nBoots);
                idsRest=1:nY; idsRest(idsTrain(5,1:d-1))=[];
                for i=1:nBoots
                    mdl = fitrsvm(X(idsTrain(5,C_(i,:)),:), Y(idsTrain(5,C_(i,:))), 'BoxConstraint', C, 'KernelFunction', 'rbf', 'Epsilon', 0.1*std(Y(idsTrain(5,C_(i,:)))), 'KernelScale', ks);
                    Ys(idsRest,i) = predict(mdl, X(idsRest,:));
                end
                QBC=zeros(1,length(idsRest));
                for i=1:length(idsRest)
                    QBC(i)=var(Ys(idsRest(i),:));
                end
                [~,idx]=max(QBC);
                idsTrain(5,d)=idsRest(idx);
            end
            
            %% 6: RSAL; the first minN samples were obtained randomly
            if d == minN
                idsTrain(6,1:d)=idsTrain(1,1:d);
            else
                mdl = fitrsvm(X(idsTrain(6,1:d-1), :), Y(idsTrain(6,1:d-1)), 'BoxConstraint', C, 'KernelFunction', 'rbf', 'Epsilon', 0.1*std(Y(idsTrain(6,1:d-1))), 'KernelScale', ks);
                Y1 = predict(mdl, X(idsTrain(6,1:d-1), :));
                % train residual model and predict residual
                mdl = fitrsvm(X(idsTrain(6,1:d-1), :), Y(idsTrain(6,1:d-1))-Y1, 'BoxConstraint', C, 'KernelFunction', 'rbf', 'Epsilon', 0.1*std(Y(idsTrain(6,1:d-1))), 'KernelScale', ks);
                Y2 = abs(predict(mdl, X(idsRest,:))); 
                [~,idx] = max(Y2);
                idsTrain(6,d)=idsRest(idx);
            end
            
           %% Compute RMSEs and CCs
            for idxAlg = 1:nAlgs  
                mdl = fitrsvm(X(idsTrain(idxAlg,1:d),:), Y(idsTrain(idxAlg,1:d)), 'BoxConstraint', C, 'KernelFunction', 'rbf', 'Epsilon', 0.1*std(Y(idsTrain(idxAlg,1:d))), 'KernelScale', ks);
                YPred = predict(mdl, XI);
                RMSEs{s}(idxAlg,r,d) = sqrt(mean((YPred-YI).^2));
                CCs{s}(idxAlg,r,d) = corr(YPred,YI);
            end
        end % end for n = minN:maxN
	end     % end for r = 1:nRepeats
    
    %% Plot results
    fig = figure('color','white');
    set(fig,'position',[100,100,1000,375]);
    sty_idx = 0;
    subplot(1,2,1); hold on;
    for i=1:nAlgs
        ii = plotAlgsIds(i);
        sty_idx = sty_idx + 1;
        plot(minN:maxN,squeeze(mean(RMSEs{s}(ii,:,minN:maxN,1),2)),'Color', Color{sty_idx}, 'LineStyle', LineStyle{sty_idx});
    end
    axis tight; box on;
    ylabel('RMSE'); xlabel('M');
    legend(plotAlgs_list,'location','northeast');
    title(datasets{s});
    
    subplot(1,2,2); hold on;
    sty_idx = 0;
    for i=1:nAlgs
        ii = plotAlgsIds(i);
        sty_idx = sty_idx + 1;
        plot(minN:maxN,squeeze(mean(CCs{s}(ii,:,minN:maxN,1),2)),'Color', Color{sty_idx}, 'LineStyle', LineStyle{sty_idx});
    end
    axis tight; box on;
    ylabel('CC'); xlabel('M');
    legend(plotAlgs_list,'location','southeast');
    title(datasets{s}); drawnow;
end % end for s = 1:nDats