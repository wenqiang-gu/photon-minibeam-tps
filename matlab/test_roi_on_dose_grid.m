function test_roi_on_dose_grid()
% Run in MATLAB: addpath('matlab'); test_roi_on_dose_grid
cg = struct('dimensions',[4 4 3],'cubeCoordOffset',[10 20 30], ...
    'direction',eye(3),'resolution',struct('x',2,'y',2,'z',2));
ct.cubeDim = [4 4 3];
cst = cell(1,6); cst{1,2} = 'Example';
original = false(4,4,3);
original(2,2,2) = true; original(2,4,2) = true;
cst{1,4} = {find(original)};
dij.ctGrid = cg; dij.doseGrid = cg;
[m,c] = roi_on_dose_grid(ct,cst,dij,'Example');
assert(isequal(m,original) && c.complete);

% Asymmetric crop offset, half CT spacing: one ROI voxel retained, one clipped.
dg = struct('dimensions',[4 4 4],'cubeCoordOffset',[11.5 19.5 31.5], ...
    'direction',eye(3),'resolution',struct('x',1,'y',1,'z',1));
dij.doseGrid = dg;
[m,c] = roi_on_dose_grid(ct,cst,dij,'Example');
expected = false(4,4,4); expected(3:4,1:2,1:2)=true;
assert(isequal(m,expected));
assert(~c.complete && c.covered_fraction == 0.5 && nnz(m)==8);
% A sparse matrix row must refer to the same MATLAB voxel as the mapped mask.
D = sparse(find(expected),ones(8,1),(1:8)',64,1);
doseCube = reshape(full(D*2),[4 4 4]);
assert(sum(doseCube(m)) == 72);

% Rigidly rotating both grids must preserve all mask associations.
a = pi/4; R = [cos(a) -sin(a) 0; sin(a) cos(a) 0; 0 0 1];
dij.ctGrid.direction = R; dij.ctGrid.cubeCoordOffset = (R*cg.cubeCoordOffset')';
dij.doseGrid.direction = R; dij.doseGrid.cubeCoordOffset = (R*dg.cubeCoordOffset')';
[m,c] = roi_on_dose_grid(ct,cst,dij,'Example');
assert(isequal(m,expected) && c.covered_fraction==0.5);

% Empty original ROI is distinct from an uncovered nonempty ROI.
cst{1,4} = {[]};
[m,c] = roi_on_dose_grid(ct,cst,dij,'Example');
assert(~any(m(:)) && isnan(c.covered_fraction));
fprintf('ROI mapping tests passed.\n');
end
