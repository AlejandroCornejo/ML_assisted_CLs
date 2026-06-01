function [RBFinputs] = RegressionViaRBF_master_weights(qM,W,DATA_interp)
%==========================================================================
% RegressionViaRBF_master_weights
%
% PURPOSE
% -------
% Offline construction of a Gaussian RBF surrogate for the MAW--ECM adaptive
% cubature weights
%
%       W = W(qM),
%
% where:
%   - qM ∈ R^{nM} are the master/latent coordinates
%     (typically nM = 2 in the present bivariate setting),
%   - W ∈ R^{nW} are the state-dependent integration weights associated with
%     the selected MAW--ECM cubature points.
%
% The routine fits a differentiable multi-output RBF model and stores all
% quantities required for online evaluation of:
%   - W(qM),
%   - dW/dqM,
%   - d²W/dqM²,
%
% through the dedicated online evaluator:
%
%       EvalWeightsRBF_Online.m
%
% This replaces the older use of EvalTauSlaveRBF_Online.m, which was designed
% for master/slave reduced coordinates and assembled a combined tau-vector.
% In the present routine, the target is not a slave coordinate qS(qM), but the
% adaptive weight vector W(qM). Therefore, no tau assembly is performed in the
% online weight evaluator.
%
% ORIGIN
% ------
% This routine derives from RegressionViaRBF_master_slave.m. The original
% routine was intended to regress slave reduced coordinates qS(qM). Here, the
% same RBF machinery is reused for the regression of MAW--ECM cubature weights.
%
% IMPORTANT: CONSTRAINTS OF THE REGRESSED OBJECT
% ----------------------------------------------
% The MAW--ECM weights are not generic regression outputs. At the sampled
% latent states they satisfy local cubature constraints, including:
%
%   (i)  positivity:
%            W_i(qM) >= 0,
%
%   (ii) volume conservation:
%            sum_i W_i(qM) = Vtot,
%
% and, more generally, local exactness/equilibrium-related constraints inherited
% from the adaptive cubature construction.
%
% The present routine performs a direct RBF regression of W. Therefore, these
% constraints are guaranteed only at the training samples, not necessarily at
% arbitrary online query points. In particular, the RBF prediction may exhibit:
%   - negative weights,
%   - loss of total-volume conservation,
%   - violation of local exactness constraints,
%   - oscillatory behavior near active-set transitions or boundaries.
%
% For this reason, this routine should be interpreted as the baseline direct
% differentiable regression layer. Additional admissibility corrections
% (rectification, normalization, constrained maps, or alternative regressions)
% should be implemented in separate evaluators or postprocessing layers if
% strict online admissibility is required.
%
% INPUTS
% ------
% qM          : Master/latent coordinates.
%               Size: [nM x Nsnap].
%               In the current bivariate case, nM = 2.
%
% W           : MAW--ECM adaptive weights at the sampled latent states.
%               Size: [nW x Nsnap].
%               Rows correspond to retained cubature points.
%               Columns correspond to latent samples.
%
% DATA_interp : Configuration structure. The relevant substructure is
%
%                   DATA_interp.RBFinputs_MAWECM
%
%               If absent, legacy defaults may be inherited from
%               DATA_interp.RBFinputs.
%
%               Main fields:
%                 .c_factor_length_parameter
%                       Factor controlling the anisotropic RBF length scales.
%
%                 .PolynomialCorrectionNumberOfMonomials
%                       0 : pure RBF,
%                       1 : RBF + constant polynomial,
%                       2 : RBF + affine polynomial in (qM_1,qM_2).
%
%                 .lambdaREGULARIZATION
%                       Tikhonov regularization parameter added to the kernel
%                       matrix.
%
%                 .RefineFactorForPlot
%                       Refinement factor used in diagnostic surface plots.
%
%                 .ExtensionFactorExtrapolationPlot
%                       Extra plotting margin around the training domain.
%
%                 .SlavesModesToBePlotted
%                       Legacy name; here it denotes the indices of weight
%                       components to be plotted.
%
% OUTPUT
% ------
% RBFinputs : Structure containing all data required by EvalWeightsRBF_Online:
%
%               .qM_train
%                    Training latent points, size [nM x Nsnap].
%
%               .l
%                    Anisotropic RBF length scales, size [nM x 1].
%
%               .Alpha
%                    RBF coefficients for normalized outputs,
%                    size [Nsnap x nW].
%
%               .Beta
%                    Polynomial coefficients,
%                    size [np x nW], where np = 0, 1, or 3.
%
%               .s
%                    RMS output scaling used for normalization,
%                    size [nW x 1].
%
%               .polyMode
%                    Polynomial correction mode: 0, 1, or 2.
%
%               .nameFunctionEvaluate
%                    Set to 'EvalWeightsRBF_Online'.
%
% OFFLINE ALGORITHM
% -----------------
% (1) Read and complete RBF configuration.
%
% (2) Compute anisotropic RBF length scales from the sampled latent grid.
%
% (3) Normalize each weight component by its RMS value:
%
%         s_i = sqrt( (1/Nsnap) * sum_k W_i(k)^2 ).
%
%     This improves conditioning when different weights have different
%     magnitudes.
%
% (4) Assemble the Gaussian RBF kernel matrix:
%
%         K_ij = exp( - ||qM_i - qM_j||_l^2 ).
%
% (5) Optionally augment the RBF approximation with a constant or affine
%     polynomial correction.
%
% (6) Solve the regularized multi-output interpolation/regression system for
%     all weight components simultaneously.
%
% (7) Store the regression coefficients and normalization data in RBFinputs.
%
% (8) Optionally generate diagnostic plots and perform a basic online
%     evaluation check using EvalWeightsRBF_Online.
%
% NUMERICAL NOTES
% ---------------
% - The quality of the online prediction is strongly affected by the smoothness
%   of the offline MAW--ECM weight fields. Active-set transitions, boundary
%   clamping, or very small weights may be difficult to regress accurately.
%
% - Increasing lambdaREGULARIZATION can improve conditioning but may also
%   introduce smoothing bias.
%
% - Increasing c_factor_length_parameter enlarges the RBF support and usually
%   produces smoother fields, at the risk of washing out localized features.
%
% - Polynomial correction may improve global trends but does not enforce
%   positivity or conservation.
%
% AUTHOR / LOG
% ------------
% JAHO, 14-Dec-2025:
%   Original master/slave RBF regression routine for qS(qM).
%
% JAHO, 07-Feb-2026:
%   Modified to regress MAW--ECM adaptive cubature weights W(qM).
%
% JAHO, Apr-2026:
%   Online evaluation redirected to EvalWeightsRBF_Online.m, avoiding the
%   master/slave tau assembly of EvalTauSlaveRBF_Online.m.
%==========================================================================

if nargin == 0
    load('tmp1.mat')
 %     DATA_interp.RBFinputs_MAWECM.ExtensionFactorExtrapolationPlot = 0 ; 
  %  DATA_interp.RBFinputs_MAWECM.PolynomialCorrectionNumberOfMonomials =2; 
%             DATA_interp.RBFinputs_MAWECM.lambdaREGULARIZATION = 1e-10 ; 
%             DATA_interp.RBFinputs_MAWECM.RefineFactorForPlot =6; 
%             DATA_interp.RBFinputs_MAWECM.c_factor_length_parameter =[1 1]; 

      

    close all
    
end



DATA_interp = DefaultField(DATA_interp,'RBFinputs_MAWECM',DATA_interp.RBFinputs) ;
 RBFinputs = DATA_interp.RBFinputs_MAWECM ; 
% ----------------------------------------------------
% STEP 2: Computing length scales for each direction
% ----------------------------------------------------

RBFinputs = DefaultField(RBFinputs,'c_factor_length_parameter',[5 5]) ;
coeffRESHAPE = DATA_interp.coeffRESHAPE_rows2_cols1 ; 
l = LengthParametersRBFn(coeffRESHAPE,qM,RBFinputs) ;

% ------------------------------------------
% STEP 3) Normalization slave amplitudes
% ------------------------------------------
% W is (Mslv x Nsnap)
Nsnap = size(W,2);
% RMS scaling (recommended)
s = sqrt(sum(W.^2, 2) / Nsnap);
% Protect against zero scale
s = max(s, 1e-14);
% Normalized slaves
Wh = bsxfun(@rdivide, W, s);     % equivalent to bsxfun(@times,W,1./s)

% STEP 4
% BUILDING THE KERNEL MATRIX
% ------------------------------------------
% STEP 4) Build Gaussian RBF Kernel matrix K
% ------------------------------------------
% qM: (2 x Nsnap)  -> master coordinates
% l : (2 x 1) or (1 x 2) -> anisotropic length scales [l1; l2]
% Gaussian kernel: K_ik = exp( - r_ik^2 ),  r_ik^2 = ((q1_i-q1_k)/l1)^2 + ((q2_i-q2_k)/l2)^2
q1 = qM(1,:);   % 1 x Nsnap
q2 = qM(2,:);   % 1 x Nsnap
l1 = l(1);
l2 = l(2);
% Pairwise differences (Nsnap x Nsnap)
D1 = bsxfun(@minus, q1.', q1);    % (i,k): q1(i) - q1(k)
D2 = bsxfun(@minus, q2.', q2);    % (i,k): q2(i) - q2(k)
% Anisotropic squared distances
R2 = (D1./l1).^2 + (D2./l2).^2;   % Nsnap x Nsnap
% Gaussian kernel matrix
K = exp(-R2);                     % Nsnap x Nsnap
% ------------------------------------------
% Truncate kernel matrix for visualization
% ------------------------------------------
PLotMATRIX = 0;
if PLotMATRIX == 1
    threshold = 1e-4;
    Kplot = K;
    Kplot(abs(Kplot) < threshold) = 0;
    figure;
    spy(Kplot);
    title('Effective support of Gaussian RBF kernel (thresholded)');
end
% ------------------------------------------
% STEP 5) Build affine (polynomial) matrix P
% ------------------------------------------
% qM: (2 x Nsnap)
% P : (Nsnap x 3) = [1, qm1, qm2]
RBFinputs = DefaultField(RBFinputs,'PolynomialCorrectionNumberOfMonomials',0) ;
polyMode = RBFinputs.PolynomialCorrectionNumberOfMonomials ;
switch polyMode
    case 0
        P = [];                       % no polynomial correction
    case 1
        P = ones(Nsnap,1);            % constant only
    case 2
        P = [ones(Nsnap,1), q1', q2'];% affine: constant + linear terms
    otherwise
        error('polyMode must be 0 (RBF), 1 (const), or 2 (affine).');
end
% ------------------------------------------
% STEP 6) Solve for coefficients: (alpha, beta)
% ------------------------------------------
% K:    (Nsnap x Nsnap)
% P:    (Nsnap x 3)
% Wh:  (Mslv x Nsnap)   normalized slave targets
% We solve for each slave: [K+lambda I, P; P', 0]*[alpha; beta] = [y; 0]
% Do it in block RHS form for all slaves at once.
RBFinputs = DefaultField(RBFinputs,'lambdaREGULARIZATION',1e-10) ;
lambda = RBFinputs.lambdaREGULARIZATION;  % choose based on conditioning/noise (start small, increase if needed)
% Regularized kernel block
Klam = K + lambda*eye(Nsnap);
Y = Wh.';                             % Nsnap x Mslv
np = size(P,2);
if np == 0
    % ---- Case 1) RBF only ----
    Alpha = Klam \ Y;         % Nsnap x Mslv
    Beta  = zeros(0, size(Y,2));  % empty (0 x Mslv)
else
    % ---- Case 2/3) RBF + polynomial (const or affine) ----
    Z = zeros(np,np);
    A_sys = [Klam, P;
        P.',  Z];                      % (Nsnap+np) x (Nsnap+np)
    
    B_sys = [Y;
        zeros(np, size(Y,2))];         % (Nsnap+np) x Mslv
    
    Xcoef = A_sys \ B_sys;
    
    Alpha = Xcoef(1:Nsnap, :);              % Nsnap x Mslv
    Beta  = Xcoef(Nsnap+1:end, :);          % np x Mslv
    
    res = norm(A_sys*Xcoef - B_sys, 'fro')/max(1,norm(B_sys,'fro'));
    fprintf('Augmented system relative residual: %.3e\n', res);
    
end
% 
% % STEP 7) GRAPHICAL ASSESSMENT
% qm = qM ;
% W_Rbf = EvalRBF_Affine_Multi(qm, qM, l, Alpha, Beta, s) ;
% %errorPREDI = norm(W_Rbf-W,'fro')/norm(W,'fro') ; % Just to check error is small

RBFinputs = DefaultField(RBFinputs,'SlavesModesToBePlotted',[]) ;

if isempty(RBFinputs.SlavesModesToBePlotted)
    ModesToBePlotted = 1:size(W,1) ;
end
RBFinputs = DefaultField(RBFinputs,'RefineFactorForPlot',2) ;
RBFinputs = DefaultField(RBFinputs,'ExtensionFactorExtrapolationPlot',0.05) ;
refineFactor = RBFinputs.RefineFactorForPlot;
extendFrac = RBFinputs.ExtensionFactorExtrapolationPlot;


DATA_interp = DefaultField(DATA_interp,'LocLegend',[]) ; 
LocLegend = DATA_interp.LocLegend ; 

LocLegend = DefaultField(LocLegend,'X','qM_1') ; 
LocLegend = DefaultField(LocLegend,'Y','qM_2') ; 
LocLegend = DefaultField(LocLegend,'Z','WEIGHTS') ; 
LocLegend = DefaultField(LocLegend,'TITLE','WEIGHTS') ; 


% DATA_interp.LocLegend.X = 'qM_1' ; 
% DATA_interp.LocLegend.Y = 'qM_2' ;
% DATA_interp.LocLegend.Z = 'bCONSTR' ;
% DATA_interp.LocLegend.TITLE = 'Constrained values vector b' ;


for imodeLOC = 1:length(ModesToBePlotted)
    iMODE = ModesToBePlotted(imodeLOC) ;
    h = PlotRBF_Affine_SurfaceW(qM, W, coeffRESHAPE, l, Alpha, Beta, s, iMODE, refineFactor, extendFrac,LocLegend) ;
end

% 
% iMODE = 9;
% hpos = PlotPositiveSoftmaxRBF_SurfaceW(qM,W,coeffRESHAPE,l,Alpha,Beta,s,iMODE,refineFactor,extendFrac,LocLegend,DATA_interp);
%DATA_interp.iMODE = 9 ; 
%hpos = PlotPositiveRectifiedRBF_AllModesW(qM,W,coeffRESHAPE,l,Alpha,Beta,s,refineFactor,extendFrac,LocLegend,DATA_interp);
PlotPositive(qM,W,coeffRESHAPE,l,Alpha,Beta,s,refineFactor,extendFrac,LocLegend,DATA_interp);
 % STEP 8) TESTING ONLINE EVALUATION
%
CHECKING_DERIVATIVES = 0 ; 
qM_Test = [0
    0] ;
if CHECKING_DERIVATIVES == 1
    
    % qM_train is 2 x Nsnap
    x = qM(1,:);
    y = qM(2,:);
    
    xq = prctile(x,25) + 0.37*(prctile(x,75)-prctile(x,25));
    yq = prctile(y,25) + 0.41*(prctile(y,75)-prctile(y,25));
    qM_Test = [xq; yq];
    
end
%   RBFinputs = DATA_interp.RBFinputs ;
RBFinputs.l = l ;
RBFinputs.qM_train = qM ;
RBFinputs.Alpha = Alpha ;
RBFinputs.Beta = Beta ;
RBFinputs.s = s ;
RBFinputs.polyMode =polyMode ;
%RBFinputs.Am =  Am ; 
%     DATA.qM_train : (2 x Nsnap) training master points (centers)
%     DATA.l        : (2 x 1) length scales [l1; l2]
%     DATA.Alpha    : (Nsnap x Mslv) RBF coefficients (normalized)
%     DATA.Beta     : (np x Mslv) polynomial coeffs (np = 0,1,3)
%     DATA.s        : (Mslv x 1) normalization scales (undo normalization)
%   Optional:
%     DATA.polyMode : 0 (RBF), 1 (const), 2 (affine) consistency check

%[W_Test,J_test,H_test] = EvalWeightsRBF_Online(qM_Test,RBFinputs);
%errorPREDI = norm(W_Test-W,'fro')/norm(W,'fro') ;

% Evaluate RBF at all training points
W_rec = EvalWeightsRBF_Online(qM,RBFinputs);

% Relative interpolation error
err_interp = norm(W_rec - W,'fro') / norm(W,'fro');

% Component-wise relative errors
err_each = vecnorm(W_rec - W,2,2) ./ max(vecnorm(W,2,2),1e-14);

fprintf('Global interpolation error = %.3e\n',err_interp);
fprintf('Max component error        = %.3e\n',max(err_each));
fprintf('Mean component error       = %.3e\n',mean(err_each));



% if CHECKING_DERIVATIVES == 1
%     [W_true, J_true, H_true] = ManufacturedTruthAtPoint(qM_Test, coeffs);
%     %[W_rbf, J_rbf, H_rbf] = EvalSlaveRBF_Online( qM_query,RBFinputs);
%     % Errors
%     fprintf('rel err value:  %.3e\n', norm(W_Test-W_true)/max(1,norm(W_true)));
%     fprintf('rel err grad:   %.3e\n', norm(J_test-J_true,'fro')/max(1,norm(J_true,'fro')));
%     fprintf('rel err hess:   %.3e\n', norm(H_test(:)-H_true(:))/max(1,norm(H_true(:))));
%     
%     abs_val  = norm(W_Test - W_true);
%     abs_grad = norm(J_test - J_true,'fro');
%     abs_hess = norm(H_test(:) - H_true(:));
%     
%     n_val  = norm(W_true);
%     n_grad = norm(J_true,'fro');
%     n_hess = norm(H_true(:));
%     
%     fprintf('abs value  : %.3e,  norm true: %.3e\n', abs_val,  n_val);
%     fprintf('abs grad   : %.3e,  norm true: %.3e\n', abs_grad, n_grad);
%     fprintf('abs hess   : %.3e,  norm true: %.3e\n', abs_hess, n_hess);
%     
% end


% DATA_evaluateTAU_and_DER.nameFunctionEvaluate =
% DATA_evaluateTAU_and_DER.RBFinputs = RBFinputs ;
RBFinputs.nameFunctionEvaluate = 'EvalWeightsRBF_Online';
