function  ECMdata = MAW_ECM_genV1(A,DATA,Wfe,DATA_interp,zINI,wINI,...
    qMASTER,LOCALdata,Uel,Qall_w,DATAoffline)
% MAW_ECM_genV1
%==========================================================================
% BIVARIATE MANIFOLD-ADAPTIVE WEIGHTS ECM (MAW-ECM)
%
%   See DOCS/MAW_ECM_genV1_DOC.tex , or DOCS/MAW_ECM_genV1_DOC.pdf
%   ! kile DOCS/MAW_ECM_genV1_DOC.tex  &
% J.A. Hernández Ortega (JAHO)
% Barcelona — Balmes 185
%
% Created:           30-Jan-2026 (Friday)
% Major update:      09-Feb-2026 (HGs Pedralbes, Barcelona)
% Current revision:  24-Apr-2026 (Barcelona)
%
%==========================================================================
% PURPOSE
% -------
% This routine constructs a sparse cubature rule with manifold-adaptive
% weights for a bivariate nonlinear homogenization problem. Starting from an
% initial ECM rule (zINI, wINI), the algorithm progressively eliminates
% integration points while redistributing the weights so as to preserve the
% exactness constraints at each sampled point of the nonlinear manifold.
%
% In contrast with classical ECM, the weights are not constant, but depend on
% the latent coordinates:
%
%       qMASTER = [qM_1 ; qM_2]
%
% which parameterize the deformation state (e.g. axial + shear components).
%
% The final outcome is:
%   • a reduced set of integration points,
%   • a weight field w(qMASTER),
%   • a regression model for online evaluation of the weights.
%
%==========================================================================
% LINEAGE AND DEVELOPMENT HISTORY
% -------------------------------
% This function is the continuation of the MAW-ECM / CECM developments:
%
%   CECM_based_ManifAdWeights_PLaCE
%
% documented in:
%
%   DOCS/DOC_CECM_based_ManifAdWeights_PLaCE.pdf
%
% That original routine addressed univariate latent manifolds (1D qMASTER)
% in inelastic problems (plasticity / damage). The algorithm combined:
%
%   (i)   local exactness constraints at each manifold sample,
%   (ii)  progressive sparsification (single-point elimination),
%   (iii) least-change redistribution of weights,
%   (iv)  active-set enforcement of nonnegativity.
%
% The corresponding basis construction was formalized in:
%
%   DOCS/DOC_MAWecmBasisMATRIX_LOCAL_noverlap.pdf
%
% and implemented in:
%
%   MAWecmBasisMATRIX_LOCAL_noverlap
%
% The pruning loop relied on two complementary stages:
%
%   • No-enforcement stage:
%         Loop_MAWecmNOENF_cl
%     (see DOCS/DOC_Loop_MAWecmNOENF_cl.pdf)
%
%   • Explicit nonnegativity stage:
%         Loop_MAWecmNOENFe
%     (see DOCS/DOC_Loop_MAWecmNOENFe.pdf)
%
% with the active-set update described in:
%
%   DOCS/DOC_nn_update_active_set.pdf
%
% -------------------------------------------------------------------------
% GRAPH-BASED EXTENSION (FEB-2026)
% --------------------------------
% Numerical evidence showed that explicit nonnegativity enforcement introduces
% strong non-smoothness (kinks, spikes) in the mapping:
%
%       qMASTER -> w(qMASTER)
%
% This led to the graph-based MAW-ECM formulation developed in:
%
%   24_GRAPH_MAWecm.mlx
%   MAW_ECM_extended.pdf
%   MAW_ECM_impl_v2.pdf
%
% where the redistribution step is regularized through a Laplacian operator
% defined on the manifold sampling graph. The corresponding implementation is:
%
%   GraphBased_MAW_ECM_2v
%
% relying on:
%
%   prune_step_optionB
%   (see DOCS/DOC_prune_step_optionB.pdf)
%
% This version introduces graph-coupled redistribution and new candidate
% ranking criteria based on smoothness/roughness metrics.
%
% -------------------------------------------------------------------------
% CURRENT EXTENSION (BIVARIATE CASE)
% ---------------------------------
% The present routine generalizes the previous univariate framework to the
% bivariate case studied in:
%
%   23_2strainsHOMOG.mlx
%   25_MAWECM_hyp2var.mlx
%
% The key structural change is that the latent space is now two-dimensional.
% The samples are assumed to lie on a structured grid defined by:
%
%   DATA_interp.coeffRESHAPE_rows2_cols1 = [ny, nx]
%
% and the graph Laplacian is constructed as a finite-element-like stiffness
% matrix over a quadrilateral mesh in the (qM_1,qM_2) plane:
%
%   [COORgraph, CNgraph] = quadMeshFromXY(QM1, QM2)
%   Kgraph = ComputeK_scl(...)
%
% This replaces the 1D chain Laplacian used in the original implementation.
%
%==========================================================================
% MATHEMATICAL SETTING
% --------------------
% Let:
%   nP  = number of candidate integration points,
%   Nm  = number of manifold samples.
%
% The adaptive weights are:
%
%   wADAPT ∈ R^{nP × Nm}
%
% where each column corresponds to one manifold state.
%
% For each manifold node j:
%
%   U{j}' * w(:,j) = b{j}
%
% must hold, together with:
%
%   w(:,j) ≥ 0   (in explicit enforcement stage)
%
% The algorithm removes rows (integration points) globally across all nodes,
% recomputing the weights so that all constraints remain satisfied.
%
%==========================================================================
% ALGORITHM OVERVIEW
% ------------------
% 1. Build fixed integration scaffold (constant function).
% 2. Construct bivariate graph Laplacian Kgraph from qMASTER.
% 3. Build orthonormal projector Q from Qall_w.
% 4. Construct local exactness operators (U, b) and initialize wADAPT.
% 5. Iteratively prune integration points:
%
%    a) Rank candidates (typically by aggregate weight).
%    b) Attempt least-change removal (no positivity enforcement).
%    c) If unsuccessful, switch to explicit enforcement:
%         - local (Loop_MAWecmNOENFe), or
%         - graph-based (GraphBased_MAW_ECM_2v).
%
% 6. Stop when:
%       • number of points ≈ number of modes, or
%       • no feasible removal exists.
%
% 7. Fit regression model:
%
%       w(qMASTER) via RBF (RegressionViaRBF_master_weights)
%
% 8. Validate accuracy of the resulting cubature rule.
%
%==========================================================================
% NUMERICAL OBSERVATIONS (APR-2026)
% ---------------------------------
% • The reduction in number of ECM points is significant (e.g. 29 → ~8),
%   while maintaining good accuracy.
%
% • The mapping qMASTER → w(qMASTER) is intrinsically irregular in the
%   bivariate case.
%
% • Graph-based smoothing improves robustness locally, but does not enforce
%   global smoothness of the weight field.
%
% • The main source of non-smoothness is the active-set enforcement of
%   nonnegativity, not the regression itself.
%
% • Given the project timeline (May 2026), priority is placed on the online
%   phase and practical reduction, rather than a full resolution of the
%   smoothness/positivity trade-off.
%
%==========================================================================
% DEPENDENCIES
% ------------
% SVDT
% DefaultField
% quadMeshFromXY
% ComputeK_scl
% MAWecmBasisMATRIX_LOCAL_noverlap
% Loop_MAWecmNOENF_cl
% Loop_MAWecmNOENFe
% GraphBased_MAW_ECM_2v
% RegressionViaRBF_master_weights
% large2smallINCLUDEREP
% CheckAccurayMAT_ECMl2
% make_weights_evolution_gif
%
%==========================================================================
% REMARK
% ------
% The correctness of the graph-based smoothing critically depends on the
% consistency between:
%   • ordering of qMASTER,
%   • structure of the latent grid,
%   • ordering of A, U, b, and wADAPT.
%
% Any mismatch will invalidate the Laplacian operator and the smoothing step.
%==========================================================================
if nargin == 0
    load('tmp1.mat')
    close all
%     LOCAL.INTERACTIVE = true;
%     LOCAL.MAKE_GIF = false;
%     LOCAL.FACE_ALPHA = 0.18;
%     LOCAL.SHOW_LEGEND = false;
%     DATA_interp.ParametersGraphMAW_ECM.NumberOfCandidatesToTry =3;
    
    %     DATA_interp.ParametersGraphMAW_ECM.CriterionUsedForRankingCandidates = 2;
    %     DATA_interp.RBFinputs_MAWECM.ExtensionFactorExtrapolationPlot = 0 ;
    %     DATA_interp.RBFinputs_MAWECM.PolynomialCorrectionNumberOfMonomials = 1;
    %     DATA_interp.RBFinputs_MAWECM.lambdaREGULARIZATION = 1e-10 ;
    %     DATA_interp.RBFinputs_MAWECM.RefineFactorForPlot =6;
    %     DATA_interp.RBFinputs_MAWECM.c_factor_length_parameter =[2 2];
    %     DATA_interp.ParametersGraphMAW_ECM.alphaSMOOTH = 20;
    %     DATA_interp.ParametersGraphMAW_ECM.SmoothLaplacianAllIterations = 0 ;
    %     DATA_interp.ParametersGraphMAW_ECM.IncrementalSmoothing = 1 ;
    %
    %      DATA_interp.RBFinputs_MAWECM.FrozenPadding_Activate = 0;
    %     DATA_interp.RBFinputs_MAWECM.FrozenPadding_NumberGhostLayers = 4;
    %     DATA_interp.RBFinputs_MAWECM.FrozenPadding_PlotAugmentedGrid = 1;
    
    %     DATA_interp.ParametersGraphMAW_ECM.UseActiveSetAwareRanking = 0;
    %
    %     DATA_interp.ParametersGraphMAW_ECM.ActiveSetAwareWeights.alphaR = 1.0;   % amplitude roughness
    %     DATA_interp.ParametersGraphMAW_ECM.ActiveSetAwareWeights.alphaJ = 1.0;   % edge-jump roughness
    %     DATA_interp.ParametersGraphMAW_ECM.ActiveSetAwareWeights.alphaZ = 0.2;   % total clamping
    %     DATA_interp.ParametersGraphMAW_ECM.ActiveSetAwareWeights.alphaB = 0.5;   % boundary clamping
    %     DATA_interp.ParametersGraphMAW_ECM.ActiveSetAwareWeights.alphaF = 0.5;   % zero-set fragmentation
    %     DATA_interp.ParametersGraphMAW_ECM.ActiveSetAwareWeights.alphaI = 0.5;   % isolated zero nodes
    %
    %     DATA_interp.ParametersGraphMAW_ECM.ActiveSetTolZero = 1e-12;
    %
    %     DATA_interp.ParametersGraphMAW_ECM.UseProgressivePruning = 0;
    %     DATA_interp.ParametersGraphMAW_ECM.ProgressivePruning_msteps = 20;
    %
    %     DATA_interp.ParametersGraphMAW_ECM.UseBoundaryWeightedKgraph  =1 ;
    %       DATA_interp.ParametersGraphMAW_ECM.BoundaryPenaltyLambda  =10 ;
    %       DATA_interp.ParametersGraphMAW_ECM.BoundaryPenaltyLength  =5 ;
    
    
    
    
end

if nargin < 12 || isempty(LOCAL)
    LOCAL.MAKE_GIF   = false;                 % if true, build GIF after loop
    LOCAL.GIF_FILE   = 'weights_evolution.gif';
    LOCAL.DELAY_TIME = 0.25;                  % seconds per frame
    LOCAL.DPI        = 120;                   % export resolution
    LOCAL.SHOW_LEGEND = true;                 % legend in the GIF frames
end

ONES = ones(size(A{1},1),1) ; % This is a vector of ones. It represents the constant function
Ufixed = SVDT(ONES) ;


% --------------------------------------------
%%%%% LAPLACIAN OF THE GRAP (fe-LIKE)
% ------------------------------------------
% Recall that this function is ad-hoc, in the sense, that
% it can be only used for manifold regression in which the
% samples lie on a cartesian grid (or structure grid)
% qMASTER is a 2 x m   matrix
% We know that
% DATA_interp.coeffRESHAPE_rows2_cols1 = [ny, nx]
% where mx*my
% How the data is structured can be seen in
% /home/joaquin/Desktop/CURRENT_TASKS/MATLAB_CODES/FE_HROM/LARGE_STRAINS/NonLinearManifolds/SubsamplingMasterSpace.m
% For instance
% % coeffRESHAPE = [ny,nx] ;
ny = DATA_interp.coeffRESHAPE_rows2_cols1(1) ;
nx = DATA_interp.coeffRESHAPE_rows2_cols1(2) ;

QM1 = reshape(qMASTER(1,:), ny, nx);
QM2 = reshape(qMASTER(2,:), ny, nx);
[COORgraph, CNgraph] = quadMeshFromXY(QM1, QM2);

PlotGRAPH = 0 ;
if PlotGRAPH == 1
    plotMesh2D_v1(COORgraph, CNgraph) ;
end

TypeElement = 'Quadrilateral' ;
% Eigen value problem
% Stiffness and mass matrix (parametric domain)

Kgraph = ComputeK_scl(COORgraph,CNgraph,TypeElement)   ;

DATA_interp.ParametersGraphMAW_ECM = DefaultField( ...
    DATA_interp.ParametersGraphMAW_ECM, ...
    'UseBoundaryWeightedKgraph',0);

DATA_interp.ParametersGraphMAW_ECM = DefaultField( ...
    DATA_interp.ParametersGraphMAW_ECM, ...
    'BoundaryPenaltyLambda',2);

DATA_interp.ParametersGraphMAW_ECM = DefaultField( ...
    DATA_interp.ParametersGraphMAW_ECM, ...
    'BoundaryPenaltyLength',1);

if DATA_interp.ParametersGraphMAW_ECM.UseBoundaryWeightedKgraph == 1
    
    lambdaB = DATA_interp.ParametersGraphMAW_ECM.BoundaryPenaltyLambda;
    ellB    = DATA_interp.ParametersGraphMAW_ECM.BoundaryPenaltyLength;
    
    Kgraph = BoundaryWeightedGraphLaplacian(Kgraph,ny,nx,lambdaB,ellB);
    
end




sqW = sqrt(Wfe) ;
Q = bsxfun(@times,Qall_w,1./sqW) ;  % Now Q'*WfeD*Q = I
Q  = SVDT(Q) ;
%  disp('Check integration error')
%  Am = cell2mat(A) ;
%
% ApproxERR = Am - Q*(Q'*Am);
% ApproxERR = norm(ApproxERR,'fro')/norm(Am,'fro') ;


[U,b,wADAPT,DATAoffline,numberMODES] = MAWecmBasisMATRIX_LOCAL_noverlap(Ufixed,zINI,qMASTER,A,DATAoffline,wINI,Wfe,Q) ;

PLOT_b = 0;
DATA_interp.LocLegend.X = 'qM_1' ;
DATA_interp.LocLegend.Y = 'qM_2' ;
DATA_interp.LocLegend.Z = 'bCONSTR' ;
DATA_interp.LocLegend.TITLE = 'Constrained values vector b' ;


if PLOT_b == 1
    
    RegressionViaRBF_master_weights(qMASTER,cell2mat(b),DATA_interp) ;
end
DATA_interp.LocLegend  = [] ;
% Now we have that, for instance,

% U{20}'*wADAPT(:,20)-b{20}
%
% ans =
%
%      0
%      0
%      0
%      0
%      0


DATAoffline = DefaultField(DATAoffline,'MaxNumberZeros_Active_Set_Loop_MAW_ECM',1) ; % = 1;
MaxZerosMAW = DATAoffline.MaxNumberZeros_Active_Set_Loop_MAW_ECM ;

%
VOL  = sum(wINI) ;
% [ hhh,htitle]= PlotAUX_mawecm(LOCAL,qMASTER,wADAPT,VOL) ;
% ---- NEW: history storage (no plotting in-loop) -------------------------
historyW = {};              % cell array; each entry is wADAPT at an iteration
historyTitle = {};          % optional: titles per iteration
% Capture the initial state before reduction starts (as "iteration 0")
historyW{end+1} = wADAPT;
historyTitle{end+1} = sprintf('Weights vs qMASTER — initial (iter 0)');


% Initialization
iCAND = 1:length(wINI);  % Candidate points to be eliminated (local indexes)
jELIM = [] ;
%USE_LEAST_NORM = 1;
k  =1;

DATAoffline= DefaultField(DATAoffline,'MaximumNumberOfECMpoints',0) ; % = 15;
DATAoffline= DefaultField(DATAoffline,'Use_GlobalGraph_MAW_ECM_2ndstage',0) ; %
DATA_interp= DefaultField(DATA_interp,'ParametersGraphMAW_ECM',[]) ; %
DATA_interp.ParametersGraphMAW_ECM= DefaultField(DATA_interp.ParametersGraphMAW_ECM,'SmoothLaplacianAllIterations',0) ; %
DATA_interp.ParametersGraphMAW_ECM= DefaultField(DATA_interp.ParametersGraphMAW_ECM,'CriterionUsedForRankingCandidates',0) ;
% 0 --- Number of points clamped to zero
% 1 ... Relative smootheness

SmoothLaplacianAllIterations = DATA_interp.ParametersGraphMAW_ECM.SmoothLaplacianAllIterations ;

% See /home/joaquin/Desktop/CURRENT_TASKS/MATLAB_CODES/TESTING_PROBLEMS_FEHROM/112_NonLIN_ROM_RBF/24_GRAPH_MAWecm.mlx

max_nmodes= max(numberMODES) ; % Maximum number of modes
max_nmodes = max(max_nmodes,DATAoffline.MaximumNumberOfECMpoints) ;
%maxIter = 20 ;
DATA_interp.ParametersGraphMAW_ECM = DefaultField( ...
    DATA_interp.ParametersGraphMAW_ECM, ...
    'UseActiveSetAwareRanking',0);

DATA_interp.ParametersGraphMAW_ECM = DefaultField( ...
    DATA_interp.ParametersGraphMAW_ECM, ...
    'UseProgressivePruning',0);
DATA_interp.ParametersGraphMAW_ECM = DefaultField( ...
    DATA_interp.ParametersGraphMAW_ECM, ...
    'ProgressivePruning_msteps',20);


while   length(iCAND) > max_nmodes
    disp(['iter = ',num2str(k),', Length initial candidate set = ',num2str(length(iCAND))])
    %     if k == 31
    %         disp('Borrar esto')
    %     end
    % 1) Deciding which points to eliminate
    RRR= sum(wADAPT,2) ;
    ilocPOS = find(RRR>0) ;  % 20-Jan-2026....Isnt't ilocPOS = iCAND always ???
    % ilocPOS are the indexes of those points which have not eliminated yet
    % Next we order such points according to which has, on average, the
    % lower weights (ascending order)
    [RRRloc_sort,indSORT] = sort(RRR(ilocPOS));
    kLOC = 1;
    if SmoothLaplacianAllIterations == 0
        [wADAPT_tent,iCAND_tent,jELIM_tent,kLOC] = Loop_MAWecmNOENF_cl(U,ilocPOS,iCAND,wADAPT,b,indSORT,jELIM);
        % if successful (kLOC below number of max. iterations), wADAPT_tent(jELIM_tent,:) = zeros (as many zeros as manifold samples)
    end
    
    if (kLOC > length(ilocPOS) && DATAoffline.MaxNumberZeros_Active_Set_Loop_MAW_ECM >0) || SmoothLaplacianAllIterations==1
        
        if SmoothLaplacianAllIterations == 0
            disp('The algorithm with no enforcement of w>0 cannot reduce anymore the number of points... ')
            disp('We switch to explicit enforcement')
        end
        
        
        if DATAoffline.Use_GlobalGraph_MAW_ECM_2ndstage == 0
            [wADAPT_tent,iCAND_tent,jELIM_tent,SOLUTION_FOUND] = ...
                Loop_MAWecmNOENFe(U,ilocPOS,iCAND,wADAPT,b,indSORT,jELIM,MaxZerosMAW);
            
        else
            if DATA_interp.ParametersGraphMAW_ECM.UseActiveSetAwareRanking == 0
                
                if DATA_interp.ParametersGraphMAW_ECM.UseProgressivePruning == 1
                    disp('Experience has shown this is unnecessary')
                    [wADAPT_tent,iCAND_tent,jELIM_tent,SOLUTION_FOUND] = ...
                        GraphBased_MAW_ECM_2v_ProgressivePruning( ...
                        U,ilocPOS,iCAND,wADAPT,b,indSORT,jELIM, ...
                        MaxZerosMAW,Kgraph,qMASTER,DATA_interp);
                else
                    
                    [wADAPT_tent,iCAND_tent,jELIM_tent,SOLUTION_FOUND] = ...
                        GraphBased_MAW_ECM_2v( ...
                        U,ilocPOS,iCAND,wADAPT,b,indSORT,jELIM, ...
                        MaxZerosMAW,Kgraph,qMASTER,DATA_interp);
                    
                end
                
                
            else
                % This has not proved effective
                error('We have found not advantage here !!! ')
                [wADAPT_tent,iCAND_tent,jELIM_tent,SOLUTION_FOUND] = ...
                    GraphBased_MAW_ECM_2v_ActiveSetAware( ...
                    U,ilocPOS,iCAND,wADAPT,b,indSORT,jELIM, ...
                    MaxZerosMAW,Kgraph,qMASTER,DATA_interp);
            end
            %
            %
            %
            %             [wADAPT_tent,iCAND_tent,jELIM_tent,SOLUTION_FOUND] = ...
            %                 GraphBased_MAW_ECM_2v(U,ilocPOS,iCAND,wADAPT,b,indSORT,jELIM,MaxZerosMAW,Kgraph,qMASTER,DATA_interp);
        end
        
        
        
        if  ~SOLUTION_FOUND
            disp('The algorithm cannot reduce the number of points anymore... ')
            break
        else
            wADAPT = wADAPT_tent ;  iCAND = iCAND_tent ; jELIM = jELIM_tent;
        end
    elseif kLOC <= length(ilocPOS)
        wADAPT = wADAPT_tent ;  iCAND = iCAND_tent ; jELIM = jELIM_tent;
    else
        disp('The algorithm cannot reduce the number of points anymore... ')
        break
    end
    
    % ---- NEW: store snapshot for this iteration (no plotting here) ------
    historyW{end+1} = wADAPT; %#ok<AGROW>
    historyTitle{end+1} = sprintf('Weights vs qMASTER — npoints %d',length(iCAND));
    
    %    hhh = SAWECMlocalPLOT_1(LOCAL,hhh,wADAPT,VOL) ;
    k = k+1;
    
end
RRR= sum(wADAPT,2) ;
ilocPOS = find(RRR>0) ;

disp(['Final number  of points = ',num2str(length(ilocPOS))])
wALL = wADAPT(ilocPOS,:) ;
setPointsALL = zINI(ilocPOS) ;

setElements_cand = large2smallINCLUDEREP(setPointsALL,DATA.MESH.ngaus) ;

DATA_interp.setElements =setElements_cand ;

DATA_interp = DefaultField(DATA_interp,'MethodForRegression_MAWECM','RadialBasisFunctions')
switch DATA_interp.MethodForRegression_MAWECM
    case 'RadialBasisFunctions'
        %   DATA_regress = MAW_ECM_regression(qMASTER,wALL,DATA_interp ) ;
        DATA_interp.LocLegend.X = 'qM_1' ;
        DATA_interp.LocLegend.Y = 'qM_2' ;
        DATA_interp.LocLegend.Z = 'w' ;
        DATA_interp.LocLegend.TITLE = 'Weight' ;
        
        DATA_interp.RBFinputs_MAWECM = DefaultField(DATA_interp.RBFinputs_MAWECM,'FrozenPadding_Activate',0);
        
        if DATA_interp.RBFinputs_MAWECM.FrozenPadding_Activate == 0
            [DATA_regress] ...
                =RegressionViaRBF_master_weights(qMASTER,wALL,DATA_interp) ;
        else
            warning('Not necessary, ignore it')
            [DATA_regress] ...
                =RegressionViaRBF_master_weights_FrozenBoundaryPadding(qMASTER,wALL,DATA_interp) ;
            
            
            
        end
        
        
    otherwise
        error('Option not implemented')
end

%
% if IS_DAMAGE_PROBLEM
%     DATA_regress.IndexLinear_damageMODEL = 1;
%     DATA_regress.IndexNonlinear_damageMODEL = 2;
% end


%end


ECMdata.wRED.DATA_regress = DATA_regress;
ECMdata.setPoints = setPointsALL ;
% ECMdata.wRED.Values = wALL ;
% ECMdata.wRED.q = qMASTER ;
ECMdata.setElements = setElements_cand ;

nlatent = size(DATA_regress.qM_train,1) ;

DATAoffline = DefaultField(DATAoffline,'Index_q_used_for_regression_WEIGHTS_SAW_ECM',1:nlatent) ;


ECMdata.wRED.IndexDOFl_q = DATAoffline.Index_q_used_for_regression_WEIGHTS_SAW_ECM;
disp(['Selected elements = ',num2str(setElements_cand(:)')])


% for   icmp = 1:size(A{1},2)
%     Acomp = zeros(length(setPointsALL),length(A)) ;
%     for icluster = 1:length(A)
%         Acomp(:,icluster) = A{icluster}(setPointsALL,icmp) ;
%     end
%
%
%     figure(3043+icmp)
%     hold on
%     title(['Evolution internal forces MAW-ECM points, comp ',num2str(icmp)])
%     xlabel('qPLAST')
%     ylabel('Internal force density')
%     for ipoints = 1:length(setPointsALL)
%         plot(qMASTER,Acomp(ipoints,:),'DisplayName',['Point = ',num2str(setPointsALL(ipoints)),' elem = ',num2str(setElements_cand(ipoints))])
%     end
%     legend show
%
%
%
% end

disp('Cheching accuracy MAW-ECM')
CheckAccurayMAT_ECMl2_2latent(ECMdata,A,Wfe,qMASTER,wALL) ;


%if LOCAL.MAKE_GIF


LOCAL.coeffRESHAPE = DATA_interp.coeffRESHAPE_rows2_cols1;

PLOT_3D = 0;
if PLOT_3D == 1
    warning('Not properly tested')
    make_weights_evolution_gif_2latent_allfields(historyW,qMASTER,VOL,LOCAL,historyTitle);
end
%end

