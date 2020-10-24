%% Implementation of the iRDM algorithm in the following paper:
%%
%% LIU Zi-Ang, JIANG Xue, WU Dong-Rui. Pool-Based Unsupervised Active Learning for Regression Using Iterative 
%% Representativeness-Diversity Maximization (iRDM). Pattern Recognition Letters, 2020.
%%
%% Compare 9 approaches on linear regression model (ridge regression)
%%
%% 1. RS (Random sampling)
%%
%% 2. P-ALICE in the paper: M. Sugiyama, S. Nakajima, "Pool-based active learning in approximate linear regression,"
%%    Machine Learning 75 (3) (2009) 249-274.
%%
%% 3. GSx in the paper: D. Wu, C-T Lin and J. Huang*, "Active Learning for Regression Using Greedy Sampling,"
%%    Information Sciences, vol. 474, pp. 90-105, 2019.
%%
%% 4. RD in the paper: D. Wu, "Pool-based sequential active learning for regression,"
%%    IEEE Trans. on Neural Networks and Learning Systems, 30(5), pp. 1348-1359, 2019.
%%
%% 5. iRDM
%%
%% 6. QBC in the paper: T. RayChaudhuri, L. G. Hamey, "Minimisation of data collection by active learning,"
%%    in: Proc. IEEE Int¡¯l. Conf. on Neural Networks, Vol. 3, Perth, Australia, 1995, pp. 1338-1341.
%%
%% 7. EMCM in the paper: W. Cai, Y. Zhang, J. Zhou, "Maximizing expected model change for active learning in regression,"
%%    in: Proc. IEEE 13th Int¡¯l. Conf. on Data Mining, Dallas, TX, 2013, pp. 51-60.
%%
%% 8. RD-EMDM in the paper: D. Wu, "Pool-based sequential active learning for regression,"
%%    IEEE Trans. on Neural Networks and Learning Systems, 30(5), pp. 1348-1359, 2019.
%%
%% 9. iGS in the paper: D. Wu, C-T Lin and J. Huang*, "Active Learning for Regression Using Greedy Sampling,"
%%    Information Sciences, vol. 474, pp. 90-105, 2019.
%%
%% Dongrui Wu, drwu@hust.edu.cn

clc; clearvars; close all; rng('default');
datasets = {'Concrete-CS','Tecator-Moisture'};
nDatasets = length(datasets);
nRepeats = 100; % Number of repeats to get statistically significant results
minN = 5;       % mininum number of samples to select
maxN = 50;      % maximum number of samples to select
Cmax = 5;       % Maximum number of iterations for iRDM; c_max in the paper
rr =  0.1;      % Ridge regression parameter
nBoots = 4;     % Parameters for EMCM and QBC
train_test_rate = 0.5;   % Train-test rate in paper

% ALR algorithms for RR model (1-5: unsupervised ALR; 6-9: supervised ALR)
Algs_list =   {'RS', 'P-ALICE', 'GSx', 'RD', 'iRDM',  'QBC', 'EMCM', 'RD-EMDM', 'iGS'};
Algs_ids =    [ 1,      2,        3,     4,    5,       6,     7,       8,        9];
plotAlgs_list = {'RS', 'QBC', 'EMCM', 'RD-EMCM', 'iGS', 'P-ALICE', 'GSx', 'RD', 'iRDM'};
plotAlgsIds =    [1,    6,      7,      8,         9,     2,        3,    4,     5];
Color =     {'k',   [1,0.7,0.5], [0.5,0.5,1], [0.5,1,0.5], [1,0.5,0.5], [0.6,0.2,0], 'b',    'g',   'r'};
LineStyle = { ':',   '--',        '--',        '--',        '--',         '-',       '-',    '-',   '-'};
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
        
        Ypre_Pool = cell(maxN, nAlgs);  % For EMCM and iGS
        
        %% 1. Random selection (RS)
        idsTrain = repmat(sort(datasample(1:nY, maxN,'replace',false)),nAlgs,1);
        
        for d = minN:maxN
            C = max(1,ceil( (d-1) * rand(nBoots,d-1) ) ); % For QBC and EMCM
            
            %% 2: P-ALICE
            bf=[ones(nY,1) X]; U=(bf'*bf)/nY; invU=inv(U+0.000001*eye(size(X,2)+1));  % add a very small number to avoid singularity
            lambdas=unique([0:.1:1 0.4:0.01:0.6]); % choose the best one from {0, .1, .2, .3, .4, .41, .42, ..., .59, .6, .7, .8, .9, 1}
            idsTrainLamdas=nan(length(lambdas),d); Q=nan(1,length(lambdas));
            n=0; bProb=zeros(1,nY);
            for lambda= lambdas
                n=n+1;
                for i=1:nY
                    bProb(i)=(sum(sum(invU.*(bf(i,:)'*bf(i,:))))).^lambda;  % probability
                end
                b = cumsum(bProb)/sum(bProb); idsTrainLamdas(n,1)=find(rand <= b,1,'first');
                for i=2:d
                    idx=idsTrainLamdas(n,1);
                    while any(idsTrainLamdas(n,1:i-1)==idx)
                        idx=find(rand <= b,1,'first');
                    end
                    idsTrainLamdas(n,i)=idx;
                end
                Xlamda=[ones(d,1) X(idsTrainLamdas(n,:),:)]; Wlamda=diag(1./bProb(idsTrainLamdas(n,:)));
                L=(Xlamda'*Wlamda*Xlamda+0.000001*eye(size(X,2)+1))\Xlamda'*Wlamda;
                Q(n)=trace(U*(L*L'));
            end
            [~,idx]=min(Q);
            idsTrain(2,1:d)= idsTrainLamdas(idx,:);
            
            %% 3: GSx
            if d==minN
                dist = mean(distX,2);  % Initialization for GSx
                [~,idsTrain(3,1)] = min(dist);
                idsRest = 1:nY; idsRest(idsTrain(3,1)) = [];
                for n = 2:minN
                    dist = min(distX(idsRest,idsTrain(3,1:n-1)),[],2);
                    [~,idx] = max(dist);
                    idsTrain(3,n) = idsRest(idx);
                    idsRest(idx) = [];
                end
            else
                idsRest = 1:nY; idsRest(idsTrain(3,1:d-1)) = [];
                dist = min(distX(idsRest,idsTrain(3,1:d-1)),[],2);
                [~,idx] = max(dist);
                idsTrain(3,d) = idsRest(idx);
            end
            
            %% 4: RD
            if d==minN
                [idsCluster,~,~,Dist]=kmeans(X,d,'MaxIter',200,'Replicates',5);
                idsP=cell(1,d);
                for n = 1:d  % find the one closest to the centroid
                    idsP{n}=find(idsCluster == n);
                    [~,idx] = min(Dist(idsP{n},n));
                    idsTrain(4,n) = idsP{n}(idx);
                end
                idsTrain_RDminN = idsTrain(4,1:d); % save for RD-EMCM
            else
                [idsCluster,~,~,Dist] = kmeans(X,d,'MaxIter',200,'Replicates',5);
                idsP=cell(1,d);
                for n = 1:d  % find the one closest to the centroid
                    idsP{n}=find(idsCluster == n);
                    [~,idx] = min(Dist(idsP{n},n));
                    idsTrain(4,n) = idsP{n}(idx);
                end
            end
            
            %% 5: iRDM
            P = idsTrain(4,1:d);
            R = nan(nY, 1);  % representaiveness
            for n = 1:nY
                R(n) = sum( distX(n, idsP{idsCluster(n)}) )/(length(idsP{idsCluster(n)})-1);
            end
            for c = 1:Cmax
                for n = 1:d
                    idsFix = P(end,:); idsFix(n) = [];
                    D = min( distX(idsP{n}, idsFix), [], 2 );   % diversity
                    [~,idx] = max(D-R(idsP{n}));
                    idsTrain(5,n) = idsP{n}(idx);
                end
                if ismember(idsTrain(5,1:d), P, 'rows')
                    break;
                else
                    P(c+1,:) = idsTrain(5,1:d);
                end
            end
            
            %% 6: QBC; the first minN samples were obtained randomly
            if d == minN
                idsTrain(6,1:d)=idsTrain(1,1:d);
            else
                Ys=repmat(Y,1,nBoots);
                idsRest=1:nY; idsRest(idsTrain(6,1:d-1))=[];
                for i=1:nBoots
                    b=ridge(Y(idsTrain(6,C(i,:))),X(idsTrain(6,C(i,:)),:),rr,0);
                    Ys(idsRest,i)=[ones(length(idsRest),1) X(idsRest,:)]*b;
                end
                QBC=zeros(1,length(idsRest));
                for i=1:length(idsRest)
                    QBC(i)=var(Ys(idsRest(i),:));
                end
                [~,idx]=max(QBC);
                idsTrain(6,d)=idsRest(idx);
            end
            
            %% 7: EMCM; the first minN samples were obtained randomly
            if d == minN
                idsTrain(7,1:d)=idsTrain(1,1:d);
            else
                Ypre_EMCM = Ypre_Pool{d-1, 7};
                Ys=repmat(Y,1,nBoots);
                idsRest=1:nY; idsRest(idsTrain(7,1:d-1))=[];
                for i=1:nBoots
                    b=ridge(Y(idsTrain(7,C(i,:))),X(idsTrain(7,C(i,:)),:),rr,0);
                    Ys(idsRest,i)=[ones(length(idsRest),1) X(idsRest,:)]*b;
                end
                EMCM=zeros(1,length(idsRest));
                for i=1:length(idsRest)
                    for j=1:nBoots
                        EMCM(i)=EMCM(i)+norm((Ys(idsRest(i),j)-Ypre_EMCM(idsRest(i)))*X(idsRest(i),:));
                    end
                end
                [~,idx]=max(EMCM);
                idsTrain(7,d)=idsRest(idx);
            end
            
            %% 8: RD-EMCM; the first minN samples were obtained by RD
            if d == minN
                idsTrain(8,1:d)=idsTrain_RDminN;
            else
                Ypre_RD_EMCM = Ypre_Pool{d-1, 8};
                Ys=repmat(Y,1,nBoots);
                idsRest=1:nY; idsRest(idsTrain(8,1:d-1))=[];
                for i=1:nBoots
                    b=ridge(Y(idsTrain(8,C(i,:))),X(idsTrain(8,C(i,:)),:),rr,0);
                    Ys(idsRest,i)=[ones(length(idsRest),1) X(idsRest,:)]*b;
                end
                EMCM=zeros(1,length(idsRest));
                for i=1:length(idsRest)
                    for j=1:nBoots
                        EMCM(i)=EMCM(i)+norm((Ys(idsRest(i),j)-Ypre_RD_EMCM(idsRest(i)))*X(idsRest(i),:));
                    end
                end
                [~,idx]=max(EMCM);
                idsTrain(8,d)=idsRest(idx);
            end
            
            %% 9: iGS; the first minN samples were obtained by GSx
            if d == minN
                idsTrain(9,1:d)=idsTrain(3,1:d);
            else
                Ypre_iGS = Ypre_Pool{d-1, 9};
                idsRest=1:nY; idsRest(idsTrain(9,1:d-1))=[];
                distY=zeros(length(idsRest),d-1);
                for i=1:d-1
                    distY(:,i)=abs(Ypre_iGS(idsRest)-Y(idsTrain(9,i))*ones(length(idsRest),1));
                end
                distXY=min(distX(idsRest,idsTrain(9,1:d-1)).*distY,[],2);
                [~,idx]=max(distXY);
                idsTrain(9,d)=idsRest(idx);
            end
            
            %% Compute RMSEs and CCs
            for idxAlg = 1:nAlgs
                w = ridge(Y(idsTrain(idxAlg,1:d)),X(idsTrain(idxAlg,1:d),:), rr, 0);
                YPred = [ones(nYI,1) XI]*w;
                RMSEs{s}(idxAlg,r,d) = sqrt(mean((YPred-YI).^2));
                CCs{s}(idxAlg,r,d) = corr(YPred,YI);
                if idxAlg>=7
                    Ypre_Pool{d, idxAlg} = [ones(nY,1) X]*w;
                end
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
end 