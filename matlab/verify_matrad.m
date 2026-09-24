function verify_matrad(filename)
% Load artifacts without requiring matRad to be installed.
% In MATLAB: addpath('matlab'); verify_matrad('runs/water-cluster/derived/result.mat')
s = load(filename);
assert(isequal(size(s.ct.cubeHU{1}), s.ct.cubeDim));
assert(numel(s.ct.x) == s.ct.cubeDim(2));
assert(numel(s.ct.y) == s.ct.cubeDim(1));
for r = 1:size(s.cst,1)
    ix = s.cst{r,4}{1};
    assert(all(ix >= 1 & ix <= prod(s.ct.cubeDim) & ix == floor(ix)));
    [y,x,z] = ind2sub(s.ct.cubeDim, ix);
    assert(isequal(sub2ind(s.ct.cubeDim,y,x,z),ix));
end
assert(numel(s.stf) >= 1);
if isfield(s,'dij')
    A = s.dij.physicalDose{1};
    assert(issparse(A));
    assert(size(A,1) == prod(s.dij.doseGrid.dimensions));
    assert(size(A,2) == s.dij.totalNumOfBixels);
    assert(all(s.dij.beamNum >= 1));
    if isfield(s,'beamlet_execution') && strcmp(s.beamlet_execution,'combined')
        assert(all(s.dij.rayNum == 0) && all(s.dij.bixelNum == 0));
        assert(numel(s.column_mapping) == size(A,2));
        for j = 1:numel(s.column_mapping)
            assert(s.column_mapping{j}.beam_index == s.dij.beamNum(j));
        end
    else
        assert(all(s.dij.rayNum >= 1));
        assert(all(s.dij.bixelNum >= 1));
    end
    w = ones(size(A,2),1); % unit source exposure per column; see indexing.md (not MU)
    cube = reshape(A*w, s.dij.doseGrid.dimensions);
    assert(all(isfinite(cube(:))) && all(cube(:) >= 0));
end
fprintf('matRad structure/index checks passed.\n');
end
