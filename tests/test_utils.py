import unittest
from housing_elements import utils
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from pandas.testing import assert_frame_equal


sites = gpd.GeoDataFrame({
    'site_id': [1, 2, 3],
    'apn': ['1', '2', '3'],
    'locapn': ['1', '2', '3'],
    'geometry': [Point(1, 1), Point(1, 2), Point(1, 3)]
}, crs='EPSG:3310').to_crs('EPSG:3857')

permits = gpd.GeoDataFrame({
    'permit_id': [1],
    'geometry': [Point(15.5, 2)],
    'permyear': ['2017'],
    'apn': ['A'],
}, crs='EPSG:3310').to_crs('EPSG:3857')

class TestUtils(unittest.TestCase):
    def test_merge_on_address_lax(self):
        # Should be able to merge points within 15 meters, even if the inputs are not in a meters projection
        merged = utils.merge_on_address(sites, permits, lax=True)

        expected = gpd.GeoDataFrame({
            'site_id': [2],
            'apn_left': ['2'],
            'locapn': ['2'],
            'geometry': pd.Series([Point(1, 2)], dtype='geometry'),
            'permit_id': [1],
            'permyear': ['2017'],
            'apn_right': ['A'],
        }, crs='EPSG:3310')

        # This roundtrip is needed because there is some rounding error from converting to WebMercator
        # and back to the California projection.
        expected = expected.to_crs('EPSG:3857').to_crs('EPSG:3310')

        assert_frame_equal(pd.DataFrame(merged), pd.DataFrame(expected))

    def test_merge_on_address_lax_too_far(self):
        sites = gpd.GeoDataFrame({
            'site_id': [1, 2, 3],
            'geometry': [Point(1, 1), Point(1, 2), Point(1, 3)]
        }, crs='EPSG:3310').to_crs('EPSG:3857')

        # This point is more than 15 meters away
        permits = gpd.GeoDataFrame({
            'permit_id': [1],
            'geometry': [Point(16.5, 2)],
        }, crs='EPSG:3310').to_crs('EPSG:3857')

        merged = utils.merge_on_address(sites, permits, lax=True)

        self.assertEqual(len(merged), 0)

    def test_calculate_pdev_for_inventory_geo(self):
        num_matches, num_sites, match_rate = utils.calculate_pdev_for_inventory(sites, permits, match_by='geo', geo_matching_lax=False)

        self.assertEqual(num_matches, 0)
        self.assertEqual(num_sites, 3)
        self.assertEqual(match_rate, 0)

    def test_calculate_pdev_for_inventory_geo_lax(self):
        num_matches, num_sites, match_rate = utils.calculate_pdev_for_inventory(sites, permits, match_by='geo', geo_matching_lax=True)

        self.assertEqual(num_matches, 1)
        self.assertEqual(num_sites, 3)
        self.assertEqual(match_rate, 1/3)

if __name__ == '__main__':
    unittest.main()
