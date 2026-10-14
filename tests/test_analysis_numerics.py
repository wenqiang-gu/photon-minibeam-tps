from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.dose_analysis import main, parser, settings_for, plot_pvdr_profile, draw_pvdr_uncertainty, plt, Run
from tools.analysis_config import load_config
from tools.analysis_io import Grid, Plane, read_grid, read_plane
from tools.analysis_numerics import (Curve, data_profile, mc_pdd, mc_profile, normalization,
                                    plane_mean, profile_normalization, pvdr, ratio, reference_value, roi_mask, ssd_shape_factor, peak_normalize)


class CalculationTests(unittest.TestCase):
    def test_peak_normalization(self):
        curve=Curve(np.array([-1.,0.,10.,200.]),np.array([999.,np.nan,20.,10.]),np.ones(4)*2)
        normalized,meta=peak_normalize(curve)
        self.assertEqual(meta['peak_depth_mm'],10)
        self.assertEqual(meta['normalization_factor'],5)
        self.assertEqual(normalized.value[2],100)
        np.testing.assert_allclose(normalized.spread,10)
        self.assertEqual(curve.value[2],20)
        for values in ([0.,0.],[np.nan,np.nan]):
            with self.assertRaises(ValueError):
                peak_normalize(Curve(np.array([0.,1.]),np.array(values)))
        x=np.array([0.,10.,100.,200.]);v=np.array([20.,100.,60.,30.])
        a,_=peak_normalize(Curve(x,v*ssd_shape_factor(x,900,1000,0)))
        b,_=peak_normalize(Curve(x,v*ssd_shape_factor(x,900,1000,100)))
        np.testing.assert_allclose(a.value,b.value)

    def test_ssd_shape_correction(self):
        depths=np.array([0.,100.,200.])
        factors=ssd_shape_factor(depths,900,1000,100)
        np.testing.assert_allclose(factors,[.9801,1.,1.016736111111111])
        np.testing.assert_allclose(ssd_shape_factor(depths,900,900,100),1)
        np.testing.assert_allclose(factors*ssd_shape_factor(depths,1000,900,100),1)
        for bad in (0,-1,np.nan,np.inf):
            with self.assertRaises(ValueError):
                ssd_shape_factor(depths,bad,1000,100)

    def test_pvdr_uncertainty_propagation(self):
        values=np.array([1.,9.,1.,10.,2.,20.,4.,30.,1.,9.,1.])
        curve=Curve(np.arange(len(values),dtype=float),values)
        summary,_=pvdr(curve,prominence_fraction=.1)
        self.assertAlmostEqual(summary['mean_peak'],20)
        self.assertAlmostEqual(summary['mean_valley'],3)
        self.assertAlmostEqual(summary['peak_sd'],10)
        self.assertAlmostEqual(summary['valley_sd'],np.sqrt(2))
        expected=(20/3)*np.sqrt((10/np.sqrt(3)/20)**2+(1/3)**2)
        self.assertAlmostEqual(summary['pvdr_uncertainty'],expected)
        scaled,_=pvdr(curve.scaled(7),prominence_fraction=.1)
        self.assertAlmostEqual(scaled['pvdr_uncertainty'],expected)
        included,_=pvdr(curve,ignore_edge_peaks=False,prominence_fraction=.1)
        self.assertNotEqual(included['pvdr_uncertainty'],expected)
        flat,_=pvdr(Curve(np.arange(11.),np.array([1.,10.,1.,10.,1.,10.,1.,10.,1.,10.,1.])),prominence_fraction=.1)
        self.assertEqual(flat['pvdr_uncertainty'],0.)
        small,_=pvdr(Curve(np.arange(9.),np.array([1.,10.,2.,20.,4.,30.,3.,12.,1.])),prominence_fraction=.1)
        self.assertIsNotNone(small['pvdr'])
        self.assertIsNone(small['pvdr_uncertainty'])
        self.assertIn('two retained valleys',small['uncertainty_failure_reason'])
        invalid,_=pvdr(Curve(np.arange(4.),np.ones(4)))
        self.assertIsNone(invalid['pvdr_uncertainty'])

    def test_pvdr_uncertainty_artists(self):
        rows=[dict(depth_mm=x,pvdr=4.,pvdr_uncertainty=se) for x,se in enumerate([1.,1.,None,2.,2.])]
        fig,ax=plt.subplots()
        try:
            draw_pvdr_uncertainty(ax,rows,'mc','blue')
            paths=ax.collections[0].get_paths()
            self.assertEqual(len(paths),2)
            self.assertEqual(set(paths[0].vertices[:,1]),{3.,5.})
            self.assertEqual(set(paths[1].vertices[:,1]),{2.,6.})
            draw_pvdr_uncertainty(ax,rows,'measurement','black')
            segments=ax.collections[1].get_segments()
            self.assertEqual(len(segments),4)
            np.testing.assert_allclose(segments[0],[[0,3],[0,5]])
        finally:
            plt.close(fig)

    def test_profile_normalization_axes_interpolation_and_transforms(self):
        coords=np.arange(-4.,5.)
        z=np.array([50.,150.])
        values=100+coords[:,None,None]**2+3*coords[None,:,None]**2+z[None,None,:]
        grid=Grid(coords,coords,z,values,np.ones_like(values))
        planes=[Plane(coords,coords,values[:,:,i].T*2,float(depth)) for i,depth in enumerate(z)]
        for axis,expected in [('x',200+2+2),('y',200+2/3+6)]:
            scale,info=profile_normalization(grid,planes,1,1,axis,2,100)
            self.assertAlmostEqual(scale,2)
            self.assertAlmostEqual(info['mc_reference_mean'],expected)
        transformed=grid.shift_xy(1,-1)
        for axis in ('x','y'):
            scale,info=profile_normalization(transformed,planes,1,1,axis,2,100)
            # Independently extract each profile after transforms, then average its window.
            means=[]
            for depth in z:
                curve,_=mc_profile(transformed,axis,depth,1,1)
                means.append(curve.value[np.abs(curve.x)<=2].mean())
            self.assertAlmostEqual(info['mc_reference_mean'],np.mean(means))
            self.assertAlmostEqual(scale,info['measurement_reference_mean']/np.mean(means))
        for width in (0,-1,np.nan,np.inf,5):
            with self.subTest(width=width), self.assertRaises(ValueError):
                profile_normalization(grid,planes,1,1,'x',width,100)
        with self.assertRaisesRegex(ValueError,'outside'):
            profile_normalization(grid,planes,1,1,'x',2,200)
        for value in (0.,np.nan):
            bad=Grid(coords,coords,z,np.full_like(values,value))
            with self.assertRaises(ValueError):
                profile_normalization(bad,planes,1,1,'x',2,100)
        sparse=Grid(np.array([-2.,2.]),coords,z,values[[0,8]])
        with self.assertRaisesRegex(ValueError,'no bin'):
            profile_normalization(sparse,planes,1,1,'x',1,100)

    def test_profile_normalization_weights_profile_bins_equally(self):
        x=y=np.arange(-1.,2.)
        data=np.array([[2.,2.,2.],[6.,np.nan,np.nan],[10.,10.,10.]])
        grid=Grid(x,y,np.array([100.]),data[:,:,None])
        plane=Plane(x,y,data.T*3,100.)
        scale,info=profile_normalization(grid,[plane],1,1,'x',1,100)
        self.assertEqual(info['mc_reference_mean'],6)
        self.assertEqual(scale,3)

    def test_pvdr_plot_features_and_means_match_calculation(self):
        curve = Curve(np.arange(9.), np.array([1.,10.,2.,20.,4.,30.,3.,12.,1.]))
        for exclude in (True, False):
            with self.subTest(exclude=exclude):
                summary, features = pvdr(curve, ignore_edge_peaks=exclude, prominence_fraction=.1)
                fig, ax = plt.subplots()
                try:
                    plot_pvdr_profile(ax, curve, summary, features, source="MC", axis="x", unit="Gy")
                    for kind in ("peak", "valley"):
                        mean_line = next(line for line in ax.lines
                                         if line.get_label().startswith(f"Mean retained {kind}"))
                        np.testing.assert_allclose(mean_line.get_ydata(), summary[f"mean_{kind}"])
                        for retained in (True, False):
                            selected = [f for f in features if f['kind']==kind and f['retained']==retained]
                            label = f"{'Retained' if retained else 'Excluded'} {kind}s"
                            collections = [c for c in ax.collections if c.get_label()==label]
                            self.assertEqual(len(collections), bool(selected))
                            if selected:
                                np.testing.assert_allclose(collections[0].get_offsets(),
                                    [[f['position_mm'], f['dose']] for f in selected])
                    self.assertEqual(len(ax.texts), summary['retained_peaks']+summary['retained_valleys'])
                    self.assertIn(f"PVDR = {summary['pvdr']:.6g}", ax.get_title())
                finally:
                    plt.close(fig)
        small = Curve(np.arange(5.), np.array([1.,5.,2.,5.,1.]))
        summary, features = pvdr(small, prominence_fraction=.1)
        fig, ax = plt.subplots()
        try:
            plot_pvdr_profile(ax, small, summary, features, source="Measurement", axis="y", unit="cGy")
            self.assertEqual(len(ax.lines), 1)
            self.assertIn('PVDR unavailable', ax.get_title())
            self.assertEqual(ax.texts[0].get_text().replace('\n', ' '), summary['failure_reason'])
            self.assertEqual(len(ax.collections), 2)
        finally:
            plt.close(fig)

    def test_reference_roi_fraction_mean_and_shared_scale(self):
        x = y = np.arange(-2., 3.)
        z = np.array([50., 100., 150.])
        base = 10+x[:, None, None]**2+y[None, :, None]**2+z[None,None,:]/10
        grid = Grid(x,y,z,base)
        planes = [Plane(x,y,3*base[:,:,i].T,float(depth)) for i,depth in enumerate(z)]
        scale, meta = normalization(grid, planes, 2, 2, .5, 100)
        self.assertAlmostEqual(scale, 3)
        self.assertEqual(meta['mc_roi_bin_counts'], [9])
        self.assertEqual(meta['measurement_roi'][1]['finite_bins'], 9)
        expected = np.mean(base[1:4,1:4,1])
        self.assertAlmostEqual(meta['mc_reference_mean'], expected)
        for depth in z:
            mc, _ = mc_profile(grid, 'x', depth,2,2)
            data = data_profile(planes[list(z).index(depth)], 'x',2,2)
            np.testing.assert_allclose(mc.scaled(scale).value,data.value)
        # The scale remains reference-derived when the other depths disagree.
        planes[0].dose *= 2
        newscale,_ = normalization(grid,planes,2,2,.5,100)
        self.assertEqual(newscale,scale)

    def test_reference_interpolation_and_errors(self):
        self.assertEqual(reference_value(np.array([50.,150.]), np.array([2.,6.]),100),4)
        with self.assertRaisesRegex(ValueError,'outside'):
            reference_value(np.array([50.,150.]),np.array([2.,6.]),0)
        with self.assertRaisesRegex(ValueError,'greater than zero'):
            reference_value(np.array([50.,150.]),np.array([0.,0.]),100)
        with self.assertRaisesRegex(ValueError,'no bin'):
            roi_mask(np.array([-1.,1.]),.1)

    def test_ratio_interpolation_threshold_and_no_extrapolation(self):
        numerator=Curve(np.array([0.,2.,4.]),np.array([2.,4.,6.]))
        denominator=Curve(np.arange(-1.,6.),np.array([1.,0.,2.,4.,4.,6.,1.]))
        result=ratio(numerator,denominator)
        self.assertTrue(np.isnan(result.value[[0,1,6]]).all())
        np.testing.assert_allclose(result.value[2:6],[1.5,1.,1.25,1.])

    def test_pvdr_discards_outer_peaks_and_outer_valleys(self):
        values=np.array([1.,10.,2.,20.,4.,30.,3.,12.,1.])
        curve=Curve(np.arange(len(values),dtype=float),values)
        summary,features=pvdr(curve,prominence_fraction=.1)
        self.assertAlmostEqual(summary['pvdr'],25/4)
        self.assertEqual(summary['retained_peaks'],2)
        self.assertEqual(summary['retained_valleys'],1)
        self.assertEqual([f['index'] for f in features if f['retained']],[3,5,4])
        scaled,_=pvdr(curve.scaled(25),prominence_fraction=.1)
        self.assertAlmostEqual(scaled['pvdr'],summary['pvdr'])
        included,_=pvdr(curve,ignore_edge_peaks=False,prominence_fraction=.1)
        self.assertAlmostEqual(included['pvdr'],18/3)
        small,_=pvdr(Curve(np.arange(5.),np.array([1.,5.,2.,5.,1.])),prominence_fraction=.1)
        self.assertIsNone(small['pvdr'])
        self.assertIsNotNone(small['failure_reason'])
