function fig = visualize_dose_check(resultFile, roiName, weights, positionLPS, outputPng)
% Orthogonal physical-coordinate checks for separate OR combined dose columns.
% addpath('matlab');
% visualize_dose_check('derived/result.mat','PTV2017fw');
% Optional: weights (default all ones), slice position [X Y Z] in LPS mm
% (default original ROI centroid), output PNG path (default no file).
% Requires MATLAB R2020a or newer. No matRad or image-processing toolbox needed.
s = load(resultFile);
D = s.dij.physicalDose{1};
cg = s.dij.ctGrid; dg = s.dij.doseGrid;
% Current project supports axial CTs. Fail explicitly for other orientations.
assert(norm(double(cg.direction)-eye(3),'fro')<1e-8 && ...
    norm(double(dg.direction)-eye(3),'fro')<1e-8, ...
    'This viewer requires axial identity-direction LPS grids.');
if nargin < 3 || isempty(weights), weights = ones(size(D,2),1); end
weights = double(weights(:));
assert(numel(weights)==size(D,2) && all(isfinite(weights) & weights>=0), ...
    'Provide one finite nonnegative exposure per saved matrix column.');
[mapped, coverage] = roi_on_dose_grid(s.ct,s.cst,s.dij,roiName);
assert(size(D,1)==numel(mapped),'Dose-grid row count mismatch.');
dose = reshape(full(D*weights),double(dg.dimensions([2 1 3])));
assert(all(isfinite(dose(:))) && all(dose(:)>=0),'Invalid dose values.');
original = false(double(s.ct.cubeDim(:)'));
r = find(strcmp(s.cst(:,2),roiName));
original(s.cst{r,4}{1}) = true;
ct = double(s.ct.cubeHU{1});
co = double(cg.cubeCoordOffset(:)'); cs = spacing(cg);
cn = double(cg.dimensions(:)');
if nargin < 4 || isempty(positionLPS)
    [iy,ix,iz] = ind2sub(size(original),find(original));
    if isempty(ix)
        positionLPS = co+(cn-1).*cs/2;
        warning('Empty original ROI; showing CT center.');
    else
        positionLPS = co+([mean(ix),mean(iy),mean(iz)]-1).*cs;
    end
end
positionLPS = double(positionLPS(:)');
assert(numel(positionLPS)==3 && all(isfinite(positionLPS)), 'Use [X Y Z] LPS mm.');
lo = co-cs/2; hi = co+(cn-0.5).*cs;
assert(all(positionLPS>=lo & positionLPS<hi),'Reference point is outside CT.');
mode = 'separate';
if isfield(s,'beamlet_execution'), mode = char(s.beamlet_execution); end
fprintf('Execution: %s; %d columns; %d nonzero weights.\n',mode,size(D,2),nnz(weights));
fprintf('ROI %s: %d/%d original voxel centers covered; %d mapped dose bins.\n', ...
    roiName,coverage.covered_original_voxel_centers,coverage.original_roi_voxels,nnz(mapped));
if ~coverage.complete, warning('Incomplete ROI coverage: unscored regions are not zero dose.'); end
fprintf('Read indexing.md/metadata.json for exposure units and scorer warnings; weights are not MU.\n');

fig = figure('Color','w','Name','Dose / ROI grid validation','Position',[80 100 1500 650]);
tiledlayout(fig,1,3,'TileSpacing','compact','Padding','compact');
maxDose = max(dose(:));
if maxDose==0, maxDose=1; end % Avoid invalid color limits for an all-zero result.
cmap = hot(256);
planes = [1 2 3; 1 3 2; 2 3 1];
labels = {'X / Left (mm)','Y / Posterior (mm)','Z / Superior (mm)'};
names = {'XY','XZ','YZ'};
for p = 1:3
    a=planes(p,1); b=planes(p,2); depth=planes(p,3);
    % Full original CT extent, display sampling no coarser than either grid.
    ds=spacing(dg); step=min(cs,ds);
    u=linspace(lo(a),hi(a),max(2,ceil((hi(a)-lo(a))/step(a))+1));
    v=linspace(lo(b),hi(b),max(2,ceil((hi(b)-lo(b))/step(b))+1));
    [U,V]=meshgrid(u,v);
    xyz=repmat(positionLPS,numel(U),1); xyz(:,a)=U(:); xyz(:,b)=V(:);
    hu=reshape(sample(ct,cg,xyz,'linear',-1000),size(U));
    [d,scored]=sample(dose,dg,xyz,'linear',0);
    d=reshape(d,size(U)); scored=reshape(scored,size(U));
    nativeContour=reshape(sample(double(original),cg,xyz,'nearest',0),size(U));
    mappedContour=reshape(sample(double(mapped),dg,xyz,'nearest',0),size(U));
    ax=nexttile; hold(ax,'on');
    gray=max(0,min(1,(hu+1000)/2000)); % Fixed CT window: -1000 to 1000 HU.
    image(ax,u,v,repmat(gray,1,1,3));
    h=imagesc(ax,u,v,d); set(h,'AlphaData',0.55*double(scored));
    colormap(ax,cmap); caxis(ax,[0 maxDose]);
    % No dose overlay outside coverage, including intentionally clipped ROIs.
    contour(ax,U,V,nativeContour,[0.5 0.5],'g-','LineWidth',1.8);
    contour(ax,U,V,mappedContour,[0.5 0.5],'m--','LineWidth',1.3);
    origin=double(dg.cubeCoordOffset(:)'); n=double(dg.dimensions(:)');
    lower=origin-ds/2; upper=origin+(n-0.5).*ds;
    if positionLPS(depth)>=lower(depth) && positionLPS(depth)<upper(depth)
        rectangle(ax,'Position',[lower(a),lower(b),upper(a)-lower(a),upper(b)-lower(b)], ...
            'EdgeColor','c','LineStyle',':','LineWidth',1.5);
    end
    plot(ax,positionLPS(a),positionLPS(b),'c+','MarkerSize',8);
    axis(ax,'equal'); xlim(ax,[lo(a) hi(a)]); ylim(ax,[lo(b) hi(b)]); set(ax,'YDir','normal');
    xlabel(ax,labels{a}); ylabel(ax,labels{b});
    title(ax,sprintf('%s slice: %s = %.3f mm',names{p},char('X'+depth-1),positionLPS(depth)));
    cb=colorbar(ax); ylabel(cb,'Dose (Gy, for supplied exposures)');
end
sgtitle({sprintf('%s | %s columns | reference LPS [%.2f %.2f %.2f] mm',roiName,mode,positionLPS), ...
    'Green: original ROI | Magenta dashed: mapped ROI | Cyan dotted: scoring boundary', ...
    'Slices, not projections. Outside scoring: CT only, unscored. Review indexing.md for scorer warnings.'}, ...
    'Interpreter','none');
if nargin>=5 && ~isempty(outputPng)
    exportgraphics(fig,outputPng,'Resolution',180);
    fprintf('Figure saved: %s\n',outputPng);
end
end

function s = spacing(g)
s=double([g.resolution.x g.resolution.y g.resolution.z]);
end

function [values,inside] = sample(cube,g,xyz,method,outside)
% Evaluate in physical coordinates; clamp within edge half-voxels rather than
% losing coverage between the first/last centers and their outer boundaries.
n=double(g.dimensions(:)');
q=(xyz-double(g.cubeCoordOffset(:)'))./spacing(g);
inside=all(q>=-0.5 & q<n-0.5,2);
q=max(0,min(q,n-1));
if strcmp(method,'nearest')
    k=floor(q+0.5)+1;
    values=cube(sub2ind(n([2 1 3]),k(:,2),k(:,1),k(:,3)));
else
    % Duplicate singleton axes so linear interpolation also supports one-slice grids.
    expanded=cube;
    for axisIndex=1:3
        if size(expanded,axisIndex)==1
            reps=[1 1 1]; reps(axisIndex)=2; expanded=repmat(expanded,reps);
        end
    end
    values=interpn(expanded,q(:,2)+1,q(:,1)+1,q(:,3)+1,'linear');
end
values=values(:); values(~inside)=outside;
end
