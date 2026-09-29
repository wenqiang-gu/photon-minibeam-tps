function test_visualize_dose_check()
% Synthetic visual smoke test; opens figures. No patient files required.
% Asymmetric CT, cropped finer dose grid, plus a partially excluded ROI.
folder=tempname; mkdir(folder);
cleanup=onCleanup(@() rmdir(folder,'s'));
cg=struct('dimensions',[6 6 4],'cubeCoordOffset',[10 20 30], ...
    'direction',eye(3),'resolution',struct('x',2,'y',2,'z',2));
dg=struct('dimensions',[8 8 4],'cubeCoordOffset',[11.5 21.5 31.5], ...
    'direction',eye(3),'resolution',struct('x',1,'y',1,'z',1));
ct.cubeDim=[6 6 4]; hu=-1000*ones(6,6,4); hu(2:5,2:4,2:3)=0;
ct.cubeHU={hu};
m=false(6,6,4); m(3:5,3:4,2:3)=true; m(6,3,2)=true;
cst=cell(1,6); cst{1,2}='Synthetic'; cst{1,4}={find(m)};
dij.ctGrid=cg; dij.doseGrid=dg;
[y,x,z]=ndgrid(1:8,1:8,1:4);
v=exp(-((x-4).^2+(y-5).^2+(z-2).^2)/4);
dij.physicalDose={sparse([v(:),v(:)/2])};
file=fullfile(folder,'result.mat');
for mode={'separate','combined'}
    beamlet_execution=mode{1};
    save(file,'ct','cst','dij','beamlet_execution');
    png=fullfile(folder,[beamlet_execution '.png']);
    fig=visualize_dose_check(file,'Synthetic',[1;0],[],png);
    assert(isfile(png));
    assert(numel(findobj(fig,'Type','image'))==6); % CT + dose in each view.
    fprintf('Inspect the %s figure for aligned contours and the excluded ROI tip.\n',beamlet_execution);
end
% All-zero dose and off-center explicit reference must also display correctly.
fig=visualize_dose_check(file,'Synthetic',[0;0],[16 27 34]);
assert(isgraphics(fig));
end
