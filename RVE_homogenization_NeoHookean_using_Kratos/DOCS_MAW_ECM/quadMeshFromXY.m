function [nodes, elems] = quadMeshFromXY(X, Y)
% quadMeshFromXY  Build a structured quadrilateral mesh from X,Y matrices.
%
% INPUT:
%   X, Y : nY x nX matrices of node coordinates
%
% OUTPUT:
%   nodes : (nX*nY) x 2 array, [x y] per node
%   elems : ((nX-1)*(nY-1)) x 4 array of node indices per quad
%           Node ordering is counterclockwise: [BL BR TR TL] 

    % Basic sizes
    [nY, nX] = size(X);
    if ~isequal(size(Y), [nY, nX])
        error('X and Y must have the same size.');
    end

    % Flatten nodes (column-major, consistent with MATLAB linear indexing)
    nodes = [X(:), Y(:)];

    % Build connectivity
    nElem = (nX-1) * (nY-1);
    elems = zeros(nElem, 4);

    e = 0;
    for j = 1:(nX-1)      % x-direction index (columns)
        for i = 1:(nY-1)  % y-direction index (rows)
            % Node ids using MATLAB's sub2ind for clarity
            nBL = sub2ind([nY, nX], i,   j);
            nBR = sub2ind([nY, nX], i,   j+1);
            nTR = sub2ind([nY, nX], i+1, j+1);
            nTL = sub2ind([nY, nX], i+1, j);

            e = e + 1;
            elems(e, :) = [nBL, nBR, nTR, nTL]; % CCW quad
        end
    end
end
