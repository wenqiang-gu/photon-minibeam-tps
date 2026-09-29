function [mask, coverage] = roi_on_dose_grid(ct, cst, dij, roiName)
% Map an exact-name ROI to dose rows without changing the original cst.
% Grids use XYZ dimensions; MATLAB cubes/linear indices use [Y,X,Z].
% Nearest CT voxel at each dose-bin center; no partial-volume weighting.
rows = find(strcmp(cst(:,2), roiName));
assert(numel(rows) == 1, 'ROI name must match exactly one cst row.');
ctGrid = dij.ctGrid;
doseGrid = dij.doseGrid;
ctDims = double(ctGrid.dimensions(:)');
doseDims = double(doseGrid.dimensions(:)');
assert(isequal(double(ct.cubeDim(:)'), ctDims([2 1 3])), ...
    'Original CT and dij.ctGrid dimensions disagree.');
ctMask = false(ctDims([2 1 3]));
indices = cst{rows,4}{1};  % First nominal scenario, original CT indices.
assert(all(indices(:) >= 1 & indices(:) <= numel(ctMask) & ...
    indices(:) == floor(indices(:))), 'Invalid cst voxel indices.');
ctMask(indices) = true;

% Affine maps zero-based grid indices to physical LPS millimeters.
ctStep = double(ctGrid.direction) * diag(spacing(ctGrid));
doseStep = double(doseGrid.direction) * diag(spacing(doseGrid));
ctOrigin = double(ctGrid.cubeCoordOffset(:));
doseOrigin = double(doseGrid.cubeCoordOffset(:));
mask = false(doseDims([2 1 3]));
[y,x] = ndgrid(0:doseDims(2)-1, 0:doseDims(1)-1);
for z = 0:doseDims(3)-1
    xyz = [x(:)'; y(:)'; repmat(z,1,numel(x))];
    ctIndex = ctStep \ (doseOrigin + doseStep*xyz - ctOrigin);
    nearest = floor(ctIndex + 0.5); % Half-open CT voxel bounds.
    valid = all(nearest >= 0 & nearest < ctDims(:), 1);
    plane = false(numel(x),1);
    q = nearest(:,valid) + 1;
    plane(valid) = ctMask(sub2ind(ctDims([2 1 3]),q(2,:),q(1,:),q(3,:)));
    mask(:,:,z+1) = reshape(plane,size(x));
end

% Coverage is measured at ORIGINAL ROI voxel centers, not dose-bin counts.
% It is a discrete coverage diagnostic, not an exact geometric volume fraction.
[iy,ix,iz] = ind2sub(ctDims([2 1 3]),find(ctMask));
doseIndex = doseStep \ (ctOrigin + ctStep*[ix'-1;iy'-1;iz'-1] - doseOrigin);
inside = all(doseIndex >= -0.5 & doseIndex < doseDims(:)-0.5,1);
coverage.roi_name = roiName;
coverage.original_roi_voxels = nnz(ctMask);
coverage.covered_original_voxel_centers = nnz(inside);
coverage.uncovered_original_voxel_centers = nnz(ctMask)-nnz(inside);
coverage.complete = all(inside);
coverage.dose_grid_roi_bins = nnz(mask);
coverage.covered_fraction = NaN;
if nnz(ctMask) > 0
    coverage.covered_fraction = nnz(inside)/nnz(ctMask);
end
end

function s = spacing(grid)
s = double([grid.resolution.x, grid.resolution.y, grid.resolution.z]);
assert(all(isfinite(s) & s > 0), 'Invalid grid spacing.');
end
