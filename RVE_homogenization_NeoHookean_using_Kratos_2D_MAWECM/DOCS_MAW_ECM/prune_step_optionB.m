function [wNEW, info] = prune_step_optionB(wOLD, Acell, bcell, K, alpha, p, opts)
%==========================================================================
% PRUNE_STEP_OPTIONB
% See DOCS/DOC_prune_step_optionB.pdf 
% One pruning step for MAW-ECM with:
%   - incremental Laplacian regularization (smooth update, "no peaks")
%   - node-dependent active sets (clamp negatives only at affected nodes)
%   - no Lagrange multipliers (nullspace parametrization)
%
% INPUTS
%   wOLD   : r x Nm matrix of previous weights (assumed feasible)
%            (rows = integration points, cols = manifold samples)
%   Acell  : either
%              (a) cell {1..Nm}, each A{j} is m x r, or
%              (b) numeric 3D array m x r x Nm
%   bcell  : either
%              (a) cell {1..Nm}, each b{j} is m x 1, or
%              (b) numeric 2D array m x Nm
%   K      : Nm x Nm symmetric Laplacian (sparse recommended)
%   alpha  : scalar >= 0
%   p      : index (1..r) of the integration point to prune (global, all nodes)
%   opts   : struct with optional fields:
%              .tol_neg        (default 1e-12) negativity tolerance
%              .tol_rank_rel   (default 1e-12) relative rank tolerance
%              .maxASiter      (default 30) max active-set iterations
%              .verbose        (default false)
%              .use_dense_H    (default false) if true, forms H as sparse kron
%
% OUTPUTS
%   wNEW : r x Nm matrix of updated weights
%   info : struct:
%            .ok       : true if converged feasible for this p
%            .reason   : string if failed
%            .Dj       : cell {Nm} logical(r,1) final clamped sets
%            .nASiter  : active-set iterations used
%
% NOTES
%   - This function assumes that K corresponds to the same ordering of the
%     manifold nodes as the columns of wOLD and indices of Acell/bcell.
%   - For speed and simplicity, we build H = I + alpha*kron(K, I_r) as sparse.
%     This is typically fine for r<=O(10) and sparse K (e.g. 1D FE chain).
% JAHO,, 23-Jan-2026, Balmes 185, Barcelona.
%==========================================================================
if nargin == 0
    load('tmp.mat')
end



% -------------------- defaults --------------------
%if nargin < 8, opts = struct(); end
%if ~isfield(opts,'tol_neg'),      opts.tol_neg = 1e-12; end
if ~isfield(opts,'tol_rank_rel'), opts.tol_rank_rel = 1e-12; end
if ~isfield(opts,'maxASiter'),    opts.maxASiter = 30; end
if ~isfield(opts,'verbose'),      opts.verbose = true; end
if ~isfield(opts,'IncrementalSmoothing'),      opts.IncrementalSmoothing = 1; end


Volume = sum(wOLD(:,1),1) ;
%--------------------------------------------------------------------------
% WHY tol_neg ∝ eps^(2/3) (NUMERICAL PRACTICE FOR ACTIVE-SET METHODS)
%
% We need a tolerance to decide when a computed weight is "truly negative"
% versus a harmless numerical artefact. Using tol_neg = O(eps) is too strict:
% in this algorithm the weights are obtained through several floating-point
% operations (QR factorizations, small linear solves, nullspace projections,
% reduced-system solves, and reconstruction z = z_p + N*y). The cumulative
% roundoff and cancellation of these operations typically produces relative
% errors much larger than eps.
%
% In constrained least-squares / active-set and QP practice, a standard and
% robust choice for inequality-violation detection is:
%
%   tol ≈ c * eps^(2/3) * (typical_scale),
%
% because eps^(2/3) is a good estimate of the *effective* numerical noise
% level after repeated projection + solve steps (it is widely used in
% feasibility tests to prevent "active-set chattering").
%
% Here, the typical physical scale of a weight is Vtot/r (volume per point),
% so we scale tol_neg accordingly:
%
%   tol_neg = c * eps^(2/3) * (Vtot/r),
%
% with a modest safety factor c (e.g. 10–100) that can be tuned.
%--------------------------------------------------------------------------
[r, Nm] = size(wOLD);

c = 10 ;
tol_neg = c * eps^(2/3) * (Volume/r);
tol_rank_rel = opts.tol_rank_rel;

% -------------------- sizes -----------------------
if ~isequal(size(K), [Nm, Nm])
    error('K must be Nm x Nm, with Nm = size(wOLD,2).');
end
if p < 1 || p > r
    error('p must satisfy 1 <= p <= r, where r = size(wOLD,1).');
end

% -------------------- unify A, b access --------------------
%--------------------------------------------------------------------------
% ACCESSOR FOR A_j (PER-MANIFOLD NODE CONSTRAINT MATRICES)
%
% We allow two equivalent storage formats for the per-node constraint matrices:
%
%   (1) Cell array:
%       Acell{j} is the local matrix A_j of size (m x r) for node j.
%       - This is flexible: each node can store its own matrix explicitly.
%       - It is convenient when A_j are constructed in a loop or vary in time.
%
%   (2) Numeric 3D array:
%       Acell is a numeric tensor A3 of size (m x r x Nm) such that
%       A3(:,:,j) == A_j.
%       - This is memory-contiguous and can be faster when repeatedly indexing.
%       - It is convenient when all A_j share the same size (m x r) and are
%         generated/assembled in batch.
%
% Regardless of the chosen storage, the rest of the algorithm should see a
% uniform interface "getA(j)" that returns A_j with size (m x r).
%
% IMPORTANT: Consistency of the third dimension is critical:
%   - Nm is the number of manifold samples and is taken from wOLD (r x Nm).
%   - Therefore, A3 must have exactly Nm slices along its 3rd dimension,
%     otherwise the node indexing between weights and constraints is inconsistent.
%--------------------------------------------------------------------------
if iscell(Acell)
    getA = @(j) Acell{j};
else
    A3 = Acell;                         % m x r x Nm
    if size(A3,3) ~= Nm
        error('If Acell is numeric, it must be m x r x Nm with Nm = size(wOLD,2).');
    end
    getA = @(j) A3(:,:,j);
end

if iscell(bcell)
    getb = @(j) bcell{j};
else
    B2 = bcell;                         % m x Nm
    if size(B2,2) ~= Nm
        error('If bcell is numeric, it must be m x Nm with Nm = size(wOLD,2).');
    end
    getb = @(j) B2(:,j);
end

% -------------------- sanity on m and r --------------------
A1 = getA(1);
m  = size(A1,1);
if size(A1,2) ~= r
    error('Dimension mismatch: size(A{1},2)=%d but size(wOLD,1)=r=%d. Ensure A columns match wOLD rows.', size(A1,2), r);
end
b1 = getb(1);
if numel(b1) ~= m
    error('Dimension mismatch: size(b{1},1) must equal size(A{1},1).');
end

% Verify all nodes consistent (cheap, and prevents subtle bugs)
for j = 2:Nm
    Aj = getA(j);
    bj = getb(j);
    if size(Aj,1) ~= m || size(Aj,2) ~= r
        error('All A{j} must be m x r with constant m and r. Node %d is %dx%d.', j, size(Aj,1), size(Aj,2));
    end
    if numel(bj) ~= m
        error('All b{j} must be length m. Node %d has length %d.', j, numel(bj));
    end
end

% -------------------- build H and g (incremental Laplacian) --------------------
N = r*Nm;
%--------------------------------------------------------------------------
% INCREMENTAL LAPLACIAN QUADRATIC FORM: CONSTRUCTION OF H AND g
%
% We work with an *incremental* regularization, i.e. the objective penalizes
% changes with respect to the previous feasible solution wOLD, not absolute
% values of the weights. The quadratic objective reads:
%
%   (1/2)‖z - zOLD‖^2
% + (alpha/2)(z - zOLD)^T (K ⊗ I_r) (z - zOLD),
%
% where:
%   - z = vec(w)     is the stacked unknown (r*Nm x 1),
%   - zOLD = vec(wOLD) is the previous solution,
%   - K is the Nm x Nm graph / FE Laplacian on the manifold,
%   - I_r is the r x r identity (independent smoothing of each weight),
%   - alpha >= 0 controls the strength of the smoothness.
%
% Expanding the quadratic form and dropping constants, the objective becomes:
%
%   (1/2) z^T H z - g^T z,
%
% with:
%   H = I + alpha (K ⊗ I_r),
%   g = H zOLD.
%
% IMPORTANT STRUCTURAL POINTS:
%   - H is symmetric positive definite (SPD) as soon as alpha >= 0.
%   - The Laplacian acts ONLY across manifold nodes, never across integration
%     points, thanks to the Kronecker product with I_r.
%   - Although H is large (r*Nm x r*Nm), it is sparse if K is sparse.
%
% For typical MAW-ECM settings (r ≈ 5–10, K tridiagonal or sparse),
% explicitly forming H as a sparse matrix is acceptable and simplifies the code.
%--------------------------------------------------------------------------
%
% Total number of unknowns
%   N = r * Nm, where:
%     r  = number of integration points kept at this pruning stage,
%     Nm = number of manifold samples.
%
% Construct H = I + alpha * (K ⊗ I_r).
% - speye(N) gives the identity on the full stacked space.
% - kron(K, speye(r)) lifts the scalar Laplacian K to a vector-valued field:
%     each integration-point weight is smoothed independently across the manifold.
%
% NOTE:
%   If Nm becomes very large or K is dense, this explicit construction should
%   be replaced by a matrix-free operator using reshaping:
%     (K ⊗ I_r) vec(Z) = vec(Z * K^T).

H = speye(N) + alpha * kron(K, speye(r));

zOLD = wOLD(:);
% g    = H * zOLD;              % for incremental formulation
% 
% zOLD = wOLD(:);

if opts.IncrementalSmoothing
    % Incremental smoothing: 0.5*(z-zOLD)'*H*(z-zOLD)
    % => objective 0.5*z'*H*z - (H*zOLD)'*z + const
    g = H * zOLD;
else
    % Absolute smoothing: 0.5*||z-zOLD||^2 + 0.5*alpha*z'*(K⊗I)*z
    % => objective 0.5*z'*H*z - (zOLD)'*z + const
    g = zOLD;
end




% -------------------- initialize node-dependent active sets --------------------
%--------------------------------------------------------------------------
% INITIALIZATION OF NODE-DEPENDENT ACTIVE SETS (OPTION B)
%
% We enforce nonnegativity (w >= 0) via an active-set strategy.
% In this implementation we adopt "Option B":
%
%   - The set of clamped (fixed-to-zero) indices is allowed to be different
%     for each manifold sample (node).
%   - When negative weights appear, we clamp ONLY at the nodes where those
%     negatives occur, rather than globally clamping that index across all nodes.
%
% This yields a less invasive correction and can preserve flexibility when
% negative weights are localized in qLATENT.
%
% DATA STRUCTURE:
%   Dj is a cell array of length Nm.
%   For each node j:
%       Dj{j} is a logical vector of length r.
%       Dj{j}(i) == true  means weight i at node j is clamped to 0.
%       Dj{j}(i) == false means weight i at node j is free (subject to equality constraints).
%
% PRUNING ENFORCEMENT:
%   The pruning candidate p must be set to zero at ALL nodes, by definition of
%   pruning an integration point from the rule. Therefore, we seed the active
%   set with:
%
%       Dj{j}(p) = true  for every j,
%
% so that w_p^(j) = 0 is enforced throughout the active-set iterations.
%
% NOTE:
%   Additional indices will be added to Dj{j} during the active-set loop when
%   computed weights at node j violate nonnegativity beyond tolerance.
%--------------------------------------------------------------------------
Dj = cell(Nm,1);
for j = 1:Nm
    dj = false(r,1);  % creates a logical vector of length r initialized to false
    dj(p) = true;             % pruned index always clamped to zero at all nodes
    Dj{j} = dj;
end

%--------------------------------------------------------------------------
% ACTIVE-SET LOOP (NODE-DEPENDENT, OPTION B)
%
% This loop enforces nonnegativity of the weights via an active-set strategy.
% At each iteration we:
%   (1) Assume the current sets Dj of clamped (fixed-to-zero) indices,
%   (2) Solve the equality-constrained, incrementally regularized problem
%       under those assumptions,
%   (3) Check whether any computed weights violate nonnegativity,
%   (4) If so, update Dj locally (node by node) and repeat.
%
% The loop terminates when no new negative weights are detected (convergence),
% or when a maximum number of iterations is reached (safety stop).
%--------------------------------------------------------------------------
ok = true;
reason = "";
nASiter = 0;

for it = 1:opts.maxASiter
    nASiter = it;
    %----------------------------------------------------------------------
    % STORAGE FOR CURRENT ACTIVE-SET ITERATION
    %
    % wp_full(:,j) :
    %   Embedded particular solution at node j (size r x 1), i.e. the unique
    %   minimum-norm solution of A_{j,F_j} w = b_j with clamped entries set to 0.
    %
    % Nj_cell{j} :
    %   Embedded nullspace basis at node j (size r x d_j), spanning the local
    %   nullspace of A_{j,F_j}. These bases will later be assembled into the
    %   global nullspace operator N.
    %
    % dj_dim(j) :
    %   Dimension of the local nullspace at node j, i.e. |F_j| - m, where
    %   m is the number of equality constraints at each node.
    %----------------------------------------------------------------------
    % Build particular solution z_p and nullspace operator Nmat
    wp_full = zeros(r, Nm);     % embedded particular solutions per node
    Nj_cell = cell(Nm,1);       % embedded nullspace blocks per node
    dj_dim  = zeros(Nm,1);      % dim per node
    
    for j = 1:Nm
        %--------------------------------------------------------------------------
        % NODE-BY-NODE PROCESSING UNDER CURRENT ACTIVE SET
        %
        % For each manifold node j, we:
        %   - extract the local constraint system A_j w^(j) = b_j,
        %   - restrict it to the currently free variables F_j,
        %   - check feasibility (enough free variables and full rank),
        %   - and prepare for computing a particular solution and a nullspace basis.
        %
        % All operations here are strictly local to node j; coupling across nodes
        % only enters later through the Laplacian term in the objective.
        %--------------------------------------------------------------------------
        
        % Local constraint matrix and right-hand side at node j
        Aj = getA(j);           % m x r
        bj = getb(j);           % m x 1
        %----------------------------------------------------------------------
        % Identify free variables at node j
        %
        % Dj{j}(i) == true  -> weight i is clamped to zero at node j
        % Dj{j}(i) == false -> weight i is free at node j
        %----------------------------------------------------------------------
        Fj = ~Dj{j};  % logical index of free variables
        nF = nnz(Fj); % number of free variables at node j
        
        %----------------------------------------------------------------------
        % Basic feasibility check:
        % We must have at least as many free variables as equality constraints.
        % Otherwise A_{j,F_j} w = b_j is overdetermined and infeasible.
        %----------------------------------------------------------------------
        if nF < m
            ok = false;
            reason = sprintf('Infeasible: node %d has |F|=%d < m=%d after clamping.', j, nF, m);
            break;
        end
        %----------------------------------------------------------------------
        % Reduced constraint matrix on the free set
        %----------------------------------------------------------------------
        AjF = Aj(:,Fj);         % m x nF
        At  = AjF.';            % nF x m
        
        %----------------------------------------------------------------------
        % Robust rank test for feasibility of A_{j,F_j} w = b_j
        %
        % We require rank(A_{j,F_j}) = m, i.e. the m constraints must be
        % linearly independent when restricted to the free variables.
        %
        % We perform the test via an economy QR of At:
        %   At = Q R,  with R of size m x m.
        % The magnitude of diag(R) reveals the numerical rank.
        %----------------------------------------------------------------------
        [~,R] = qr(At, 0);       % Q not needed for rank
        diagR = abs(diag(R));
        %     % Defensive check (should not happen unless At is empty or corrupted)
        if isempty(diagR)
            ok = false;
            reason = sprintf('Infeasible: empty R at node %d.', j);
            break;
        end
        %----------------------------------------------------------------------
        % Relative rank tolerance:
        % A diagonal entry is considered nonzero if it is larger than
        % tol_rank_rel times the largest diagonal entry of R.
        %----------------------------------------------------------------------
        rtol = tol_rank_rel * max(diagR);
        rnk  = sum(diagR > rtol);
        
        if rnk < m
            ok = false;
            reason = sprintf('Infeasible: rank(A_{%d,F})=%d < m=%d.', j, rnk, m);
            break;
        end
        
        %----------------------------------------------------------------------
        % MINIMUM-NORM PARTICULAR SOLUTION ON THE FREE SET
        %
        % We seek a particular solution w_{p,j} of:
        %
        %     A_{j,F_j} w = b_j,
        %
        % with minimum Euclidean norm. This choice is crucial because:
        %   - it avoids introducing unnecessary oscillations,
        %   - it cleanly separates the constrained part from the nullspace part,
        %   - it leads to a well-posed reduced problem once combined with the
        %     Laplacian regularization.
        %
        % The minimum-norm solution is given explicitly by:
        %
        %     w_{p,j} = A_{j,F_j}^T (A_{j,F_j} A_{j,F_j}^T)^{-1} b_j,
        %
        % which is valid whenever rank(A_{j,F_j}) = m.
        %----------------------------------------------------------------------
        Gm = AjF * AjF.';        % Gram matrix, size m x m (SPD under rank condition)
        wpj = AjF.' * (Gm \ bj); % particular solution on free variables (nF x 1)
        
        %----------------------------------------------------------------------
        % NULLSPACE BASIS ON THE FREE SET
        %
        % We now construct a basis for the nullspace:
        %
        %     ker(A_{j,F_j}) = { v : A_{j,F_j} v = 0 },
        %
        % whose dimension is (nF - m). Any feasible solution at node j can be written as:
        %
        %     w^(j) = w_{p,j} + N_j y_j,
        %
        % where columns of N_j form a basis of ker(A_{j,F_j}).
        %
        % Preferred method:
        %   - Compute a full QR factorization of At = A_{j,F_j}^T.
        %   - The first m columns of Q span range(At),
        %   - The remaining columns span ker(A_{j,F_j}).
        %
        % This approach is numerically stable and cheap for the small sizes involved.
        %
        % Fallback:
        %   - If MATLAB returns an economy-size Q (or for very old versions),
        %     we fall back to null(AjF), which uses an SVD internally.
        %----------------------------------------------------------------------
        try
            [Qfull,~] = qr(At);  % typically returns nF x nF for small dense
            if size(Qfull,2) ~= nF
                error('qr returned economy Q unexpectedly');
            end
            Nj = Qfull(:, m+1:end);  % nF x (nF-m)
        catch
            Nj = null(AjF);          % nF x (nF-m) if rank = m
        end
        
        %----------------------------------------------------------------------
        % EMBEDDING LOCAL SOLUTIONS INTO THE FULL r-DIMENSIONAL SPACE
        %
        % Up to this point, both the particular solution w_{p,j} and the nullspace
        % basis N_j have been computed only on the free-variable subset F_j.
        % We now embed them back into the full r-dimensional space corresponding
        % to all integration points, inserting zeros at clamped indices.
        %
        % This embedding is essential to:
        %   - keep a uniform r-dimensional representation across all nodes,
        %   - allow straightforward assembly of the global stacked vectors/matrices,
        %   - enforce w_i^(j) = 0 automatically for all clamped indices i ∈ D_j.
        %----------------------------------------------------------------------
        %
        % Embed the particular solution:
        %   wpj_full is an r x 1 vector such that:
        %     - wpj_full(F_j) contains the minimum-norm solution wpj,
        %     - wpj_full(D_j) = 0 for clamped indices.
        wpj_full = zeros(r,1);
        wpj_full(Fj) = wpj;
        %----------------------------------------------------------------------
        % Embed the nullspace basis:
        %   Nj_full is an r x d_j matrix (d_j = nF - m) whose columns span the
        %   nullspace of A_{j,F_j}, embedded in the full r-dimensional space.
        %
        % Rows corresponding to clamped indices are identically zero, ensuring
        % that any nullspace correction preserves the clamping constraints.
        %----------------------------------------------------------------------
        Nj_full = sparse(r, size(Nj,2));
        if ~isempty(Nj)
            Nj_full(Fj,:) = Nj;
        end
        %----------------------------------------------------------------------
        % Store node-wise quantities for later global assembly
        %----------------------------------------------------------------------
        wp_full(:,j) = wpj_full;
        Nj_cell{j}   = Nj_full;
        dj_dim(j)    = size(Nj_full,2);
    end
    %----------------------------------------------------------------------
    % EXIT FROM NODE LOOP IF INFEASIBILITY WAS DETECTED
    %
    % If any node j became infeasible (e.g. not enough free variables or
    % rank-deficient reduced constraints), we immediately abort the active-set
    % iteration. The current pruning candidate is then declared invalid.
    %----------------------------------------------------------------------
    if ~ok
        break;
    end
    
    %----------------------------------------------------------------------
    % GLOBAL PARTICULAR SOLUTION
    %
    % Stack all node-wise particular solutions wp_full(:,j) column-wise into
    % a single global vector:
    %
    %     z_p = vec( [ w_{p,1}, w_{p,2}, ..., w_{p,Nm} ] ),
    %
    % of size (r*Nm) x 1.
    %
    % This vector satisfies all equality constraints A z = b as well as the
    % current clamping conditions, but ignores smoothness across nodes.
    %----------------------------------------------------------------------
    z_p = wp_full(:);
    
    %----------------------------------------------------------------------
    % TOTAL DIMENSION OF THE GLOBAL NULLSPACE
    %
    % Each node j contributes a local nullspace of dimension:
    %     d_j = |F_j| - m.
    %
    % Since the constraints are block-diagonal across nodes, the global
    % nullspace dimension is simply the sum of the local dimensions.
    % This value determines the size of the reduced system to be solved next.
    %----------------------------------------------------------------------
    total_d = sum(dj_dim);
    %----------------------------------------------------------------------
    % CHECK FOR DEGENERATE CASE: NO REMAINING NULLSPACE
    %
    % If the total nullspace dimension is zero, it means that for every node j:
    %   |F_j| = m,
    % i.e. the number of free variables exactly matches the number of equality
    % constraints. In this situation:
    %
    %   - The local systems A_{j,F_j} w = b_j are square and invertible,
    %   - The particular solution w_{p,j} is unique,
    %   - There is no remaining degree of freedom to adjust weights.
    %
    % Consequently, the global solution is fully determined by the equality
    % constraints and the clamping conditions alone, and no reduced optimization
    % problem needs to be solved.
    %----------------------------------------------------------------------
    if total_d == 0
        % No nullspace left: constraints + clamping uniquely determine z
        z = z_p;
    else
        %----------------------------------------------------------------------
        % ASSEMBLY OF THE GLOBAL NULLSPACE OPERATOR Nmat
        %
        % We now assemble the global nullspace matrix Nmat such that:
        %
        %     z = z_p + Nmat * y,
        %
        % where:
        %   - z_p is the global particular solution (already stacked),
        %   - y collects all nullspace coordinates across all nodes,
        %   - Nmat is a block-diagonal matrix with one block per node.
        %
        % IMPORTANT:
        %   - Each node j contributes a local nullspace basis Nj_full of size r x d_j,
        %     where d_j = dj_dim(j) = |F_j| - m.
        %   - Different nodes may have different d_j, hence the block widths vary.
        %   - There is no coupling between nullspaces of different nodes at this stage.
        %----------------------------------------------------------------------
        %
        % Preallocate Nmat as a sparse matrix of size (N x total_d), where:
        %   N = r * Nm              : total number of unknowns,
        %   total_d = sum_j d_j     : total nullspace dimension.
        %
        % The estimate of nonzeros is conservative and only affects allocation speed.
        %----------------------------------------------------------------------
        Nmat = spalloc(N, total_d, total_d * min(r, max(1,r-m))); %#ok<*SPRIX>
        col0 = 0;    % running column offset in Nmat
        for j = 1:Nm
            dj = dj_dim(j);  % local nullspace dimension at node j
            if dj == 0
                continue;  % no contribution from this node
            end
            %------------------------------------------------------------------
            % Global row indices corresponding to node j in the stacked vector z
            %------------------------------------------------------------------
            rows = (j-1)*r + (1:r);
            %------------------------------------------------------------------
            % Global column indices in Nmat reserved for node j's nullspace
            %------------------------------------------------------------------
            cols = col0 + (1:dj);
            % Local embedded nullspace basis at node j (r x d_j)
            Nj_full = Nj_cell{j};
            %------------------------------------------------------------------
            % Insert Nj_full into Nmat at the appropriate block location.
            % We do this via (row, col, value) triplets to preserve sparsity.
            %------------------------------------------------------------------
            [rr,cc,vv] = find(Nj_full);
            if ~isempty(vv)
                Nmat = Nmat + sparse(rows(rr), cols(cc), vv, N, total_d);
            end
            % Advance column offset for the next node
            col0 = col0 + dj;
        end
        %----------------------------------------------------------------------
        % REDUCED PROBLEM IN THE NULLSPACE
        %
        % Having parameterized all feasible solutions as:
        %
        %     z = z_p + Nmat * y,
        %
        % where:
        %   - z_p satisfies all equality constraints and clamping conditions,
        %   - columns of Nmat span the global nullspace of the constraints,
        %
        % we now determine the nullspace coefficients y by minimizing the quadratic
        % objective in this reduced space.
        %----------------------------------------------------------------------
        
        %----------------------------------------------------------------------
        % RIGHT-HAND SIDE FOR THE REDUCED SYSTEM
        %
        % Recall that the full quadratic objective is:
        %
        %     (1/2) z^T H z - g^T z,
        %
        % with g = H*zOLD (incremental formulation).
        %
        % Substituting z = z_p + Nmat*y and keeping only terms depending on y yields:
        %
        %     (1/2) y^T (Nmat^T H Nmat) y - y^T Nmat^T (g - H z_p).
        %
        % Hence the reduced right-hand side is:
        %----------------------------------------------------------------------
        rhs_full = g - H*z_p;
        
        %----------------------------------------------------------------------
        % REDUCED (NULLSPACE) SYSTEM
        %
        % Hr is the Hessian of the reduced problem:
        %   Hr = Nmat^T H Nmat,
        %
        % which is symmetric positive definite (SPD) because:
        %   - H is SPD,
        %   - Nmat has full column rank.
        %
        % fr is the reduced linear term.
        %----------------------------------------------------------------------
        Hr = (Nmat.' * H) * Nmat;    % SPD
        fr = Nmat.' * rhs_full;
        
        %----------------------------------------------------------------------
        % SOLVE FOR NULLSPACE COEFFICIENTS
        %
        % Since Hr is small (dimension total_d x total_d) and SPD, a direct solve
        % is efficient and numerically robust.
        %----------------------------------------------------------------------
        y = Hr \ fr;
        %----------------------------------------------------------------------
        % RECONSTRUCT THE FULL SOLUTION
        %
        % Add the optimal nullspace correction to the particular solution to obtain
        % the full stacked vector z, which:
        %   - satisfies all equality constraints,
        %   - respects all clamping conditions,
        %   - minimizes the incremental Laplacian-regularized objective.
        %----------------------------------------------------------------------
        z = z_p + Nmat*y;
    end
    
    w = reshape(z, r, Nm);
    
    %----------------------------------------------------------------------
    % ACTIVE-SET UPDATE BASED ON NONNEGATIVITY VIOLATIONS
    %
    % After solving the equality-constrained reduced problem for the current
    % clamping pattern Dj, we must enforce the inequality constraints:
    %
    %     w_i^(j) >= 0   for all i, j.
    %
    % We do this with a node-dependent active-set strategy ("Option B"):
    %   - If a weight becomes negative at node j, we clamp ONLY that index at
    %     that specific node (i.e., we set Dj{j}(i)=true), and re-solve.
    %   - We do NOT clamp the same index globally across all nodes, because that
    %     is more invasive and can unnecessarily reduce flexibility elsewhere.
    %
    % Termination criterion:
    %   - If no new negative weights are detected (changed=false), the active set
    %     has converged and the current solution is acceptable (up to tolerance).
    %----------------------------------------------------------------------
    changed = false; % flag: did we add any new clamped indices this iteration?
    nneg_total = 0;  % diagnostic counter: total number of negative entries detected
    
    for j = 1:Nm
        % Only inspect currently free variables at node j.
        % (Clamped entries are, by construction, enforced to be exactly zero.)
        Fj = ~Dj{j};
        % Identify indices that violate nonnegativity beyond tolerance.
        % The tolerance avoids reacting to tiny negative roundoff.
        neg_idx = find(Fj & (w(:,j) < -tol_neg));
        % Accumulate diagnostics
        nneg_total = nneg_total + numel(neg_idx);
        if ~isempty(neg_idx)
            % Enlarge the node-local active set:
            % these indices will be fixed to zero at node j in the next solve.
            Dj{j}(neg_idx) = true;   %
            % Mark that the active set changed, forcing another iteration.
            changed = true;
        end
    end
    
    if opts.verbose
        fprintf('AS iter %d: total_d=%d, negatives=%d, changed=%d\n', it, total_d, nneg_total, changed);
    end
    
    %----------------------------------------------------------------------
    % CONVERGENCE CHECK FOR THE ACTIVE-SET LOOP
    %
    % If no new negative weights were detected in this iteration, the node-local
    % active sets Dj have converged. At this point:
    %   - All equality constraints are satisfied (by construction of z = z_p + N*y),
    %   - All clamping conditions are satisfied (Dj enforced in the local solves),
    %   - All remaining free weights are nonnegative up to the chosen tolerance.
    %
    % We then accept the solution and perform a final numerical cleanup:
    %   - Set tiny values (|w| < tol_neg) to exactly zero to remove roundoff noise,
    %   - Clip any remaining small negative entries (typically at the level of
    %     floating-point roundoff) to zero.
    %
    % NOTE:
    %   This clipping is not intended to replace the active-set enforcement;
    %   it is only a final safeguard against negligible numerical artefacts.
    
    if ~changed
        % Converged active set. Small negative roundoff can be clipped.
        wNEW = w;
        % Hard-zero tiny magnitudes (removes +/- roundoff oscillations)
        wNEW(abs(wNEW) < tol_neg) = 0;
        % Enforce strict nonnegativity (should only affect tiny negatives)
        wNEW(wNEW < 0) = 0;
        % Report success and return
        info = struct('ok', true, 'reason', "", 'Dj', {Dj}, 'nASiter', nASiter);
        return;
    end
end

%----------------------------------------------------------------------
% FAILURE EXIT: INFEASIBILITY OR NO CONVERGENCE WITHIN ITERATION CAP
%
% If execution reaches this point, the active-set loop did not return a
% converged feasible solution. This can happen in two main scenarios:
%
%   (1) Infeasibility was detected (ok=false) during the node loop, e.g.:
%       - too few free variables at some node (|F_j| < m), or
%       - rank(A_{j,F_j}) < m (constraints become singular once clamping is applied).
%
%   (2) The loop hit the maximum number of active-set iterations without
%       stabilizing the active sets (changed remained true each iteration),
%       which typically indicates:
%       - the pruning candidate is not compatible with nonnegativity, or
%       - tolerances are too strict / ill-conditioning is severe.
%
% In case (2), ok may still be true (no explicit infeasibility), so we mark
% it as failure here and provide a diagnostic reason.
%
% Output policy on failure:
%   - We return wNEW = wOLD (unchanged), so downstream code cannot
%     accidentally use a partially feasible intermediate result.
%   - info contains the failure reason and the last active sets Dj.
%----------------------------------------------------------------------
if ok
    ok = false;
    reason = sprintf('Active-set did not converge within maxASiter=%d.', opts.maxASiter);
end

wNEW = wOLD;
info = struct('ok', ok, 'reason', reason, 'Dj', {Dj}, 'nASiter', nASiter);
end
