function example = dose_optimization_example(resultFile, roiName, weights)
% Minimal data-access example; does not implement an optimizer.
% addpath('matlab');
% example = dose_optimization_example('projects/my-project/derived/result.mat','PTV2017fw');
% Hand over metadata.json and indexing.md beside result.mat for units/mapping.
s = load(resultFile);
D = s.dij.physicalDose{1}; % Sparse; first nominal scenario.
assert(issparse(D), 'Expected a sparse influence matrix.');
fprintf('Available ROIs:\n');
disp(s.cst(:,2));
[roiMask, coverage] = roi_on_dose_grid(s.ct,s.cst,s.dij,roiName);
assert(numel(roiMask) == size(D,1), 'Dose row/grid mismatch.');
roiRows = find(roiMask); % These indices can directly select D rows.
fprintf('%s: %d dose bins; %d/%d original ROI voxel centers covered.\n', ...
    roiName,nnz(roiMask),coverage.covered_original_voxel_centers,coverage.original_roi_voxels);
if ~coverage.complete
    warning('ROI is incompletely scored. ROI objectives/statistics exclude its uncovered portion.');
end
if isempty(roiRows)
    warning('No dose bins map to this ROI; do not build an objective from an empty region.');
end
if nargin < 3
    weights = ones(size(D,2),1); % Illustration only; NOT a prescribed treatment.
end
weights = double(weights(:));
assert(numel(weights)==size(D,2) && all(isfinite(weights) & weights>=0), ...
    'Supply one finite nonnegative exposure per matrix column.');
% Point source: primary photons. Phase space: original accelerator histories.
% Read metadata.json/indexing.md for this project's convention; neither is MU.
% Combined jobs: one coefficient per beam, not independently adjustable bixels.
dose = full(D*weights);
example.D = D;
example.roiRows = roiRows;
example.roiMask = roiMask;
example.coverage = coverage;
example.weights = weights;
example.doseCube = reshape(dose,size(roiMask)); % [Y,X,Z], on scoring grid.
example.roiDose = dose(roiRows);
example.doseGrid = s.dij.doseGrid;
% An optimizer can use D(roiRows,:) and d_roi = D(roiRows,:)*weights.
% Retain D as sparse. Do not resample its columns to the coarser CT grid.
end
