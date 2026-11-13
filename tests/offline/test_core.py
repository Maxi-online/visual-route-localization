import unittest
from app.vloc.geo import ll_to_enu, enu_to_ll
from app.vloc.confidence import ConfidenceEstimator
from app.vloc.smoothing import EwmaSmoother

class LocalizationPrimitivesTests(unittest.TestCase):
    def test_geo_round_trip(self):
        result=enu_to_ll(ll_to_enu(55.001,37.002,55,37),55,37)
        self.assertAlmostEqual(result[0],55.001);self.assertAlmostEqual(result[1],37.002)
    def test_confidence_limits(self):
        estimator=ConfidenceEstimator()
        self.assertEqual(estimator.compute_visual(0,-1,0,0),0);self.assertEqual(estimator.compute_visual(0,2,0,0),100)
    def test_smoother_bounds(self):
        smoother=EwmaSmoother();smoother.update(0,0,100);lat,lng=smoother.update(1,1,100)
        self.assertTrue(0<lat<1);self.assertTrue(0<lng<1)
    def test_reset(self):
        smoother=EwmaSmoother();smoother.update(0,0,0);smoother.reset();self.assertEqual(smoother.update(1,2,50),(1,2))
