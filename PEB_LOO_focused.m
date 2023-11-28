%% PEB - 
clear;clc;
spm('Defaults','EEG');
%% Collate DCMs into a GCM file
% ---------------------------------------------------------------
% 
loadpaths

% Set-up GCM containing all estimated DCMs
% E.g.

GCM = {
    
% Controls      
    [filepath 'file1_modelDMN.mat'];
    [filepath 'file2_modelDMN.mat'];
    [filepath 'file3_modelDMN.mat']; 

% UWS pet0     
    [filepath 'file4_modelDMN.mat'];
    [filepath 'file5_modelDMN.mat'];
    [filepath 'file6_modelDMN.mat'];
    
            };

%% 
% Specify PEB model settings 
M = struct();
M.alpha = 1;
M.beta  = 16;
M.hE    = 0;
M.hC    = 1/16;
M.Q     = 'all';
M.maxit  = 256;
N = size(GCM,1);

% Specify design matrix for N subjects. It should start with a constant column
avg = ones(N,1);

% % % Control vs. UWS pet0
% E.g.
difference(1:3) = zeros(3,1); %edit according to group size
difference(4:6) = ones(3,1);

M.X = [avg,difference']; 
M.Xnames = {'mean','difference'}; 

% % Choose field
field = {'A'};

% Estimate model
[PEB, GCM_Updated] = spm_dcm_peb(GCM,M,field); 

%%        
save('PEB_DMN_UWS_pet0.mat','PEB');

%% 
% Search over nested PEB models.

[BMA, BMR] = spm_dcm_peb_bmc(PEB(1)); % prune away any parameters from the PEB which don't contribute
                                      % to the model evidence.
                                      % BMR: To see which connections were switched on or off in the model
                                      

%BMA = spm_dcm_bmc_peb(PEB(1));% Instead,asks which combination of connectivity parameters provides 
                               % the best estimate of between-subject effects (e.g. age or 
                               % clinical scores). In other words, it scores every combination
                               % of connectivity structures and regressors
                               % (covariates). This, if interest lies in
                               % covariates (instead of connections). 
%% 
% Review results
spm_dcm_peb_review(BMA,GCM);

%%
% Cross-validation

%------- DMN --------
% Control vs. UWS(pet0)

% Set the field to the connections the cross-validation is based on
field = {'A{1,1}(3,2)'}; % single parapeter 
% field = {'A{1,1}(3,1)','A{1,1}(3,2)',... % Subset
%     'A{1,2}(1,3)','A{1,2}(2,3)',...
%     'A{1,3}(2,1)','A{1,3}(1,2)'};

% % Data-driven LOOCV for UWS pet+
% field = {'A{1,2}(1,4)'}; 

%%
% Cross-validation
% M.X       - second level design matrix, where X(:,1) = ones(N,1) [default]
% field     - parameter fields in DCM{i}.Ep to optimise [default: {'A','B'}]
%             'All' will invoke all fields
% 
% qE        - posterior predictive expectation (group effect)
% qC        - posterior predictive covariances (group effect)
% Q         - posterior probability over unique levels of X(:,2)

[qE,qC,Q] = spm_dcm_loo(GCM,M,field); %

%----------------------------------------------------------     

%% LOSOCV - Leave-one-state-out

loadpaths;

% Original classifications
TEST = {
        %UWS  
%pet1
[filepath5 'FileA_modelDMN.mat'];          
[filepath5 'FileB_modelDMN.mat'];
[filepath5 'FileC_modelDMN.mat'];
[filepath5 'FileD_modelDMN.mat'];
[filepath5 'FileE_modelDMN.mat'];   
       };
  

% TRAIN data 
GCM =  {

% Controls            
    ['FileF_modelDMN.mat'];
    ['FileG_modelDMN.mat'];
    ['FileH_modelDMN.mat']; 
   %.... etc. 

% %pet0          
    [filepath 'file4_modelDMN.mat'];
    [filepath 'file5_modelDMN.mat'];
    [filepath 'file6_modelDMN.mat'];
   %.... etc.

       };

%%   
% Specify design matrix for N subjects.
N = size(GCM,1);
avg = ones(N,1);

difference(1:3) = zeros(3,1); % Ctrl
difference(4:6) = ones(3,1); % UWS pet0

%%

M.X = [avg,difference']; 
M.Xnames = {'mean','difference'};

Y = [1 0]; % Design matrix for test data
X = M.X; 
TRAIN = GCM; 
iX = 2; % column of desing matrix to predict

% Leave one out scheme
Ns = length(TEST); 
for i = 1:Ns
%     TRAIN(i) = []; %remove data from training for LOO participant
%     TRAIN(i+9) = []; 
    
    [Ep,Cp,P] = spm_dcm_ppd(TEST(i),TRAIN,Y,X,field); % TEST,TEST0,TEST1
    qE(i)     = Ep;
    qC(i)     = Cp;
    Q(:,i)    = P;
    
    TRAIN = GCM; %Set training data back to starting point
end

% save('DMNLOO.mat'); 


%%
% Accuracy & Confusion matrix
L = length(Q);
c = zeros(L,1);
n = 1;
r = 1;

% This when test data = UWS pet1 -- Binary precision
n = 1;

while n < 6
        if Q(1,n) > 0.5
           c(n,1) = 1;
        else
           c(n,1) = 0;
        end       
n = n + 1;
end

% Assing numbers of true/false positives and negatives
C = zeros(2,1);

C(1,1) = sum(c(:,1) == 1); 
C(2,1) = sum(c(:,1) == 0);

% Binary Precision
Tr = C(1,1); 
Tot = sum(C(:)); 

Acc = Tr/Tot;
fprintf('Accuracy = %4.2f\n', Acc);  

 
%%
% close all;
save('DMN_LOSCV_posterior.mat');

%%
% This for test = train (equal in size), UWS all
% n = 1:length(Q);
r = 1;

for n = 1:length(Q);
    if n <= 11
        if Q(1,n) > 0.5
           c(n,1) = 1;
        else
          c(n,1) = 0;
        end
        
    elseif n >= 12
        if Q(1,n) < 0.5 
           c(r,2) = 1;
        else
           c(r,2) = 0;
        end   
        if n < 17
        r = r + 1;
        end
    end
    n = n+1;
end

% Assing numbers of true/false positives and negatives
C = zeros(2,2);

C(1,1) = sum(c(:,1) == 1); 
C(1,2) = sum(c(:,1) == 0);
C(2,1) = sum(c(:,2) == 0);
C(2,2) = sum(c(:,2) == 1); 
      
% Accuracy
Tr = C(1,1) + C(2,2);
Tot = sum(C(:)); 

Acc = Tr/Tot;
fprintf('Accuracy = %4.2f\n', Acc);  

%% Save
save('DMN_Loo_ctrl.mat');

%% AUC 
% Multiple versions because of unequal sample size

% This for LOOCV - ctrl / UWS pet0 
figure; 
    tlabels = [];
    tlabels(:,:) = zeros(1,11);
    tlabels(:,12:17) = ones(1,6);
    
%     tlabels(:,2) = ones(1,6);
%     tlabels = num2cell(tlabels,2);
    
    posclass = 0;
 
    scores = [];
%     scores(1:10) = Q %%% Change to single vector
    scores(:,:) = Q(1,1:11);
    scores(:,12:17) = Q(1,12:end);
%     scores = num2cell(scores,2); 
    
    [X,Y,T,AUC] = perfcurve(tlabels,scores,posclass); %,'xvals','all');

%     subplot(4,2,d)
    plot(X,Y)
    
str = sprintf('AUC: %.2f',AUC);
text(0.5,0.2,str,'Color','k','FontSize',15)

%% AUC 
% This for LOOCV - ctrl / MCS+ 
figure; 
    tlabels = [];
    tlabels(:,:) = zeros(1,11);
    tlabels(:,12:23) = ones(1,12);
    
    posclass = 0;
 
    scores = [];
%     scores(1:10) = Q %%% Change to single vector
    scores(:,:) = Q(1,1:11);
    scores(:,12:23) = Q(1,12:end);
%     scores = num2cell(scores,2); 
    
    [X,Y,T,AUC] = perfcurve(tlabels,scores,posclass,'xvals','all');

%     subplot(4,2,d)
    plot(X,Y)
    
str = sprintf('AUC: %.2f',AUC);
text(0.5,0.2,str,'Color','k','FontSize',15)


%% AUC 
% This for MCS+ / UWS pet-
figure; 
    tlabels = [];
    tlabels(:,:) = zeros(1,12);
    tlabels(:,13:18) = ones(1,6);
    
    posclass = 0;
 
    scores = [];
    scores(:,:) = Q(1,:);
    
    [X,Y,T,AUC] = perfcurve(tlabels,scores,posclass); %,'xvals','all');

%     subplot(4,2,d)
    plot(X,Y)
    
str = sprintf('AUC: %.2f',AUC);
text(0.5,0.2,str,'Color','k','FontSize',15)


%% Accuracy
% This for ctrl / UWS pet-
% n = 1:length(Q);
r = 1;

for n = 1:length(Q);
    if n <= 11
        if Q(1,n) > 0.5
           c(n,1) = 1;
        else
          c(n,1) = 0;
        end
        
    elseif n >= 12
        if Q(1,n) < 0.5 
           c(r,2) = 1;
        else
           c(r,2) = 0;
        end   
        if n < 18
        r = r + 1;
        end
    end
    n = n+1;
end

% Assing numbers of true/false positives and negatives
C = zeros(2,2);

C(1,1) = sum(c(:,1) == 1); 
C(1,2) = sum(c(:,1) == 0);
C(2,1) = sum(c(1:6,2) == 0);
C(2,2) = sum(c(1:6,2) == 1); 
      
% Accuracy
Tr = C(1,1) + C(2,2);
Tot = sum(C(:)); 

Acc = Tr/Tot;
fprintf('Accuracy = %4.2f\n', Acc);  

%%
close all;
save('DMN_Loo_ctrl_fropar.mat');
%%
% This for UWS pet1
% n = 1:length(Q);
r = 1;

for n = 1:length(Q);
    if n <= 11
        if Q(1,n) > 0.5
           c(n,1) = 1;
        else
          c(n,1) = 0;
        end
        
    elseif n >= 12
        if Q(1,n) < 0.5 
           c(r,2) = 1;
        else
           c(r,2) = 0;
        end   
        if n < 18
        r = r + 1;
        end
    end
    n = n+1;
end

% Assing numbers of true/false positives and negatives
C = zeros(2,2);

C(1,1) = sum(c(:,1) == 1); 
C(1,2) = sum(c(:,1) == 0);
C(2,1) = sum(c(1:5,2) == 0);
C(2,2) = sum(c(1:5,2) == 1); 
      
% Accuracy
Tr = C(1,1) + C(2,2);
Tot = sum(C(:)); 

Acc = Tr/Tot;
fprintf('Accuracy = %4.2f\n', Acc);  


%%
% This for mcs+ uws pet-
% n = 1:length(Q);
r = 1;

for n = 1:length(Q);
    if n <= 12
        if Q(1,n) > 0.5
           c(n,1) = 1;
        else
          c(n,1) = 0;
        end
        
    elseif n > 12
        if Q(1,n) < 0.5 
           c(r,2) = 1;
        else
           c(r,2) = 0;
        end   
        if n < 18
        r = r + 1;
        end
    end
    n = n+1;
end

% Assing numbers of true/false positives and negatives
C = zeros(2,2);

C(1,1) = sum(c(1:12,1) == 1); 
C(1,2) = sum(c(1:12,1) == 0);
C(2,1) = sum(c(1:6,2) == 0);
C(2,2) = sum(c(1:6,2) == 1); 
      
% Accuracy
Tr = C(1,1) + C(2,2);
Tot = sum(C(:)); 

Acc = Tr/Tot;
fprintf('Accuracy = %4.2f\n', Acc);  

%%
% This for ctrl mcs+
% n = 1:length(Q);
r = 1;

for n = 1:length(Q);
    if n <= 11
        if Q(1,n) > 0.5
           c(n,1) = 1;
        else
          c(n,1) = 0;
        end
        
    elseif n > 11
        if Q(1,n) < 0.5 
           c(r,2) = 1;
        else
           c(r,2) = 0;
        end   
        if n < 23
        r = r + 1;
        end
    end
    n = n+1;
end

% Assing numbers of true/false positives and negatives
C = zeros(2,2);

C(1,1) = sum(c(1:11,1) == 1); 
C(1,2) = sum(c(1:11,1) == 0);
C(2,1) = sum(c(1:12,2) == 0);
C(2,2) = sum(c(1:12,2) == 1); 
      
% Accuracy
Tr = C(1,1) + C(2,2);
Tot = sum(C(:)); 

Acc = Tr/Tot;
fprintf('Accuracy = %4.2f\n', Acc); 

