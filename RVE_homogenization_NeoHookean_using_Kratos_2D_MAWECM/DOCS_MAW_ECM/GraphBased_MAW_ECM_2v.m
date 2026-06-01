function [wADAPT,iCAND,jELIM,FEASIBLE_SOLUTION_FOUND] = ...
    GraphBased_MAW_ECM_2v(U,ilocPOS,iCAND,wADAPT,b,indSORT,jELIM,MaxZerosMAW,Kgraph,qLATENT,DATA_interp)
%==========================================================================
% GraphBased_MAW_ECM_2v
% See DOCS/DOC_GraphBased_MAW_ECM_2v.pdf
%
% PURPOSE
% -------
% Graph-coupled greedy sparsification step for MAW-ECM (2 variables / multi-node),
% where each pruning trial is re-solved by a *single graph-coupled* quadratic step:
%   • exactness constraints are enforced node-wise:      A_j * w_j = b_j,
%   • nonnegativity is enforced with node-wise active sets (Option B),
%   • manifold smoothness is promoted through a Laplacian Kgraph acting across
%     the manifold nodes (columns of w).
%
% Relative to the classical MAW-ECM exhaustive elimination baseline
% (Loop_MAWecmNOENFe), this routine replaces “re-solve each node independently”
% by a coupled solve that discourages sharp variations of the weights over the
% latent graph during each trial elimination.
%
% -------------------------------------------------------------------------
% KEY UPDATE: CANDIDATE SELECTION BY GRAPH ROUGHNESS (ANTI-"NEEDLE" HEURISTIC)
% -------------------------------------------------------------------------
% In addition to feasibility, candidates can be ranked/selected using *roughness*
% of the *incremental* change induced by pruning:
%
%     DeltaW = w_trial(rows_kept,:) - w_current(rows_kept,:)    (rows_kept=iCAND_new)
%
% Roughness is measured on *graph edges* extracted from the Laplacian:
%
%     [Iedge,Jedge] = find(triu(-Kgraph,1));
%
% where each edge e connects the manifold nodes (Iedge(e), Jedge(e)).
% The discrete jump of DeltaW across each edge is:
%
%     Jumps(:,e) = DeltaW(:,Iedge(e)) - DeltaW(:,Jedge(e))
%
% Two spike-sensitive roughness indicators are recorded per feasible candidate:
%   • R95 : 95th percentile of |Jumps| over all kept rows and edges (robust),
%   • Rmax: max of |Jumps| over all kept rows and edges (strict spike detector).
%
% Optionally, a quadratic (L2) roughness energy can also be recorded:
%   • S = sum(sum( DeltaW .* (DeltaW*Kgraph) ))  (graph Dirichlet energy)
%
% These metrics allow rejecting/avoiding “needle” solutions (localized punctures)
% that can pass feasibility but are visually/physically undesirable.
%
% Candidate ranking criterion is controlled by:
%   DATA_interp.ParametersGraphMAW_ECM.CriterionUsedForRankingCandidates
%       0 : legacy criterion  -> minimize NumberSampleManifoldWithZeroClamped
%       1 : smoothness energy -> minimize S (graph L2 energy of DeltaW)
%       2 : roughness (robust)-> minimize R95, tie-break by Rmax (anti-needle)
%
% -------------------------------------------------------------------------
% INPUTS
% ------
% U          : 1 x Nm cell. U{j} supplies the exactness operator for node j.
%              Used to build A{j} = U{j}(ilocPOS,:)' at each pruning trial.
%
% ilocPOS    : Vector of global row indices currently eligible for elimination.
%              All pruning trials are restricted to this subset.
%
% iCAND      : Current active global indices (rows currently kept nonzero).
%              If a trial is committed, one index is removed and appended to jELIM.
%
% wADAPT     : Current feasible weights (|zINI| x Nm), satisfying:
%                wADAPT >= 0  and  A_j*wADAPT(ilocPOS,j) = b_j  for all j,
%              up to tolerances internal to prune_step_optionB.
%
% b          : Node-wise RHS for exactness constraints (cell {1..Nm} or m x Nm).
%
% indSORT    : Order in which local candidates p (indices within ilocPOS) are tested.
%
% jELIM      : Column vector of already eliminated global indices.
%              After committing, all jELIM rows are explicitly set to zero.
%
% MaxZerosMAW: Legacy parameter kept for interface compatibility (not a hard reject).
%
% Kgraph     : Nm x Nm graph Laplacian on manifold nodes (column adjacency).
%              Must be consistent with wADAPT column ordering, b, and U{j}.
%
% qLATENT    : Latent coordinates (bookkeeping/diagnostics; not used in the core).
%
% DATA_interp: Struct with parameters (defaults applied):
%              .ParametersGraphMAW_ECM.alphaSMOOTH          (default 0.1)
%              .ParametersGraphMAW_ECM.NumberOfCandidatesToTry
%                     (default length(ilocPOS))
%              .ParametersGraphMAW_ECM.IncrementalSmoothing (default 1)
%              .ParametersGraphMAW_ECM.CriterionUsedForRankingCandidates (0/1/2)
%
% -------------------------------------------------------------------------
% OUTPUTS
% -------
% wADAPT  : Updated weights. If a candidate is committed, wADAPT is replaced by
%           the corresponding feasible trial, and all rows in jELIM are set to 0.
%
% iCAND   : Updated active set (one global index removed if committed).
%
% jELIM   : Updated elimination list (appends the committed global index).
%
% FEASIBLE_SOLUTION_FOUND :
%           True if at least one candidate trial produced a feasible solution
%           (all nodes satisfy constraints and nonnegativity), in which case one
%           candidate is selected and committed. False otherwise (no changes).
%
% -------------------------------------------------------------------------
% ALGORITHM (one exhaustive pruning pass; at most one commit)
% ----------------------------------------------------------
% 1) For each local candidate p = indSORT(kLOC) (index within ilocPOS):
%      - indMINglo = ilocPOS(p)             (global pruned row)
%      - iCAND_new = setdiff(iCAND,indMINglo)
%      - Build A{j} = U{j}(ilocPOS,:)' for all nodes j
%      - Call prune_step_optionB on wADAPT(ilocPOS,:) with prune index p to obtain
%        a feasible trial wADAPT_new(ilocPOS,:) (or reject if infeasible).
%      - If feasible, record:
%           * trial weights and updated active set,
%           * clamp diagnostics (nodes with any clamp-to-zero),
%           * roughness metrics (S, R95, Rmax) computed from DeltaW on graph edges.
%
% 2) Select among feasible trials according to CriterionUsedForRankingCandidates
%    (0/1/2 described above), and commit that single elimination.
%
% 3) Commit:
%      Update iCAND, append indMINglo to jELIM, set wADAPT(jELIM,:) = 0.
%
% -------------------------------------------------------------------------
% DEPENDENCIES
% ------------
% • prune_step_optionB
% • DefaultField
%
% -------------------------------------------------------------------------
% BEHAVIOR / CONVENTIONS
% ----------------------
% • At most one elimination is committed per call (single pass).
% • An outer driver should call this function repeatedly until
%   FEASIBLE_SOLUTION_FOUND == false.
% • Diagnostics refer to the *local* prune index p in ilocPOS.
%==========================================================================
% Comments updated on 9-Feb-2026, HGs Pedralbes, Barcelona.
% JAHO

%==========================================================================

if nargin == 0
    load('tmp2.mat')
end

%kLOC = 1;
WeightsHistory = {} ;
CandidatesHistory = {} ;
NumberZeroWeigth = []  ;
indMINglo_hst = [] ;
NumberSampleManifoldWithZeroClamped = [] ;
WeightsClampedToZero = cell(size(U)) ;
IndexWeightsClampedToZero = {};
% MaxGradient = [] ;
% Norm2Gradient = [] ;

%DATA_interp.ParametersGraphMAW_ECM.alphaSMOOTH = 0.1;
DATA_interp.ParametersGraphMAW_ECM = DefaultField(DATA_interp.ParametersGraphMAW_ECM,'alphaSMOOTH',0.1) ;

alphaSMOOTH = DATA_interp.ParametersGraphMAW_ECM.alphaSMOOTH ;

NumberOfCandidatesToTry = length(ilocPOS)  ;

DATA_interp.ParametersGraphMAW_ECM = DefaultField(DATA_interp.ParametersGraphMAW_ECM,'NumberOfCandidatesToTry',NumberOfCandidatesToTry) ;
NumberOfCandidatesToTry = DATA_interp.ParametersGraphMAW_ECM.NumberOfCandidatesToTry ;
DATA_interp.ParametersGraphMAW_ECM = DefaultField(DATA_interp.ParametersGraphMAW_ECM,'IncrementalSmoothing',1) ;

% Edge list of the latent graph (undirected, no duplicates)
% See https://chatgpt.com/share/6988728c-41f4-8013-99be-12b3e9ca233b
[Iedge, Jedge] = find(triu(-Kgraph,1));   % Iedge,Jedge are node indices (1..Nm)


kLOC = 1;
SmoothScore = [] ;
Rmax_hist = [];
R95_hist  = [];
E_hist    = [];   % optional (your SmoothScore energy)


while kLOC <= length(ilocPOS)  && length(CandidatesHistory) < NumberOfCandidatesToTry
    
    
    %for kLOC =2:2 % Here we test all the possible combinations
    %    warning('borrar esto')
    indMIN = indSORT(kLOC) ;
    indMINglo = ilocPOS(indMIN) ;
    iCAND_new = setdiff(iCAND,indMINglo) ; % = [] ;
    
    EXIT_WHILE_cluster = 0 ;
    wADAPT_new = wADAPT ;
    % NzerosCLUSTER= zeros(length(U),1) ;
    
    
    p =indMIN;  % prune candidate index
    disp('---------------------------------------------')
    disp(['Candidate to be prunned (local numbering) = ',num2str(p),' (out of ',num2str(length(ilocPOS)),')' ])
    disp('---------------------------------------------')
    
    opts =[] ;
    opts.IncrementalSmoothing = DATA_interp.ParametersGraphMAW_ECM.IncrementalSmoothing ;
    A = cellfun(@(B) B(ilocPOS,:)', U, 'UniformOutput', false);
    [wADAPT_new(ilocPOS,:), info] = prune_step_optionB(  wADAPT(ilocPOS,:), A, b, Kgraph, alphaSMOOTH, p, opts);
    
    %DeltaW = wADAPT_new(iCAND_new,:) - wADAPT(iCAND_new,:);
    
    if ~info.ok
        fprintf('Candidate rejected: %s\n', info.reason);
        EXIT_WHILE_cluster = 1;
    else
        %
        rows   = iCAND_new;                              % kept rows after pruning
        USE_total_as_criterion = 1; 
        if USE_total_as_criterion ==0
        DeltaW = wADAPT_new(rows,:) - wADAPT(rows,:);    % incremental change
        else
          DeltaW = wADAPT_new(rows,:) ; %- wADAPT(rows,:); 
        end
        
        % Edge jumps (spike detector)
        Jumps = DeltaW(:,Iedge) - DeltaW(:,Jedge);       % (#rows_kept) x (#edges)
        absJ  = abs(Jumps(:));
        
        Rmax = max(absJ);
        R95  = prctile(absJ,95);
        
        % Optional quadratic graph energy (your current "SmoothScore")
        S = sum( DeltaW .* (DeltaW*Kgraph), 'all' );
        
        
        
        
        
        fprintf('Candidate accepted\n');
        %  S = sum( DeltaW .* (DeltaW * Kgraph) , 'all' );   % scalar, fast
        
        [LocWeightsClampedToZero,SampleManifoldWithZeroClamped]=  find( wADAPT_new(iCAND_new,:)==0 ) ;
    end
    
    
    
    %     for icluster = 1:length(U)
    %         Uloc = U{icluster}(iCAND_new,:) ;
    %         [~,SSS] = SVDT(Uloc) ;
    %         if   length(SSS) == size(Uloc,2)
    %             w_before= wADAPT(iCAND_new,icluster) ;
    %             if USE_exact_method == 0
    %                 [ w_after,SOLUTION_fOUND] = nn_update_active_set(Uloc, b{icluster}, w_before) ;
    %             else
    %                 % Not recommended, it produces noisy solutions
    %                 [w_after,SOLUTION_fOUND] = nn_update_lsqlin(Uloc, b{icluster}, w_before) ;
    %
    %             end
    %             IndZeros = find(w_after==0) ;
    %             WeightsClampedToZero{icluster} = IndZeros ;
    %             NzerosCLUSTER(icluster) = length(IndZeros) ;
    %             if ~SOLUTION_fOUND
    %                 % disp(['No feasible solution found, exiting, icluster = ',num2str(icluster)])
    %                 EXIT_WHILE_cluster = 1 ;
    %                 break
    %
    %             end
    %             wADAPT_new(iCAND_new,icluster) = w_after ;
    %         else
    %             disp('Uloc is not full rank, exiting')
    %             EXIT_WHILE_cluster = 1 ;
    %             break
    %
    %         end
    %     end
    
    
    if EXIT_WHILE_cluster == 0
        indMINglo_hst(end+1) = indMINglo ;
        %   NumberZeroWeigth(end+1) = max(NzerosCLUSTER) ;
        %    SampleManifoldWithZeroClamped = find(NzerosCLUSTER>0) ;
        NumberSampleManifoldWithZeroClamped(end+1) = length(SampleManifoldWithZeroClamped) ; %find(NzerosCLUSTER>0) ;
        
        %   LocWeightsClampedToZero =    WeightsClampedToZero(SampleManifoldWithZeroClamped) ;
        LocWeightsClampedToZero = unique(LocWeightsClampedToZero) ;
        SmoothScore(end+1) = S;
        disp(['Smooth score = ',num2str(S)])
        
        
        Rmax_hist(end+1) = Rmax;
        R95_hist(end+1)  = R95;
        %   E_hist(end+1)    = E;
        
        IndexWeightsClampedToZero{end+1} = iCAND_new(LocWeightsClampedToZero) ;
        
        WeightsHistory{end+1} = wADAPT_new; %(iCAND_new,:) ;
        CandidatesHistory{end+1} = iCAND_new ;
        % Compute gradient
        %         dw = diff(wADAPT_new')';
        %         dw_dq = bsxfun(@times,dw',1./Delta_q')';
        %         max_grad = max(max(abs(dw_dq))) ;
        %         MaxGradient(end+1) = max_grad ;
        %          ndw_dq = sum(sum(dw_dq.^2)) ;
        %          Norm2Gradient(end+1) = sqrt(ndw_dq)  ;
    end
    
    kLOC = kLOC + 1;
end



if isempty(CandidatesHistory) % || minNZ > MaxZerosMAW
    FEASIBLE_SOLUTION_FOUND = false;
    disp('Feasible solution not found')
else
    FEASIBLE_SOLUTION_FOUND = true ;
    disp('Feasible solution    found')
    % Max criterion
    %  [nZZ,indZZ_max] = min(MaxGradient)  ;
    
    % We choose that minimizing the number of zero weights produced during
    % interations
    
    if DATA_interp.ParametersGraphMAW_ECM.CriterionUsedForRankingCandidates ==0
        [nZZ,indZZ] = min(NumberSampleManifoldWithZeroClamped)  ;
        disp(['Weigths clamped to zero during iterations for chosen candidate = ',num2str(nZZ)])
        
        
    elseif DATA_interp.ParametersGraphMAW_ECM.CriterionUsedForRankingCandidates ==1
        [MinSmoothScore, indZZ] = min(SmoothScore);
        disp(['MinSmoothScore =',num2str(MinSmoothScore) ,'; Candidate =',num2str(indZZ)] )
        
    elseif DATA_interp.ParametersGraphMAW_ECM.CriterionUsedForRankingCandidates ==2
        % 1) best robust roughness
        minR95 = min(R95_hist);
        cand   = find(R95_hist <= minR95*(1+1e-12) + 1e-30);
        
        % 2) tie-break by max jump
        [~,k2] = min(Rmax_hist(cand));
        cand   = cand(k2);
        
        % 3) optional tie-break by energy
        % [~,k3] = min(E_hist(cand)); cand = cand(k3);
        
        indZZ = cand;
        
        
    else
        error('Option not implemented')
    end
    
    iCAND = CandidatesHistory{indZZ} ;
    indMINglo = indMINglo_hst(indZZ) ;
    jELIM = [jELIM; indMINglo] ;
    wADAPT_new = WeightsHistory{indZZ} ;
    wADAPT_new(jELIM,:) = 0  ;
    wADAPT = wADAPT_new ;
    %   break
end