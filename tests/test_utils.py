import unittest
from housing_elements import utils
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from pandas.testing import assert_frame_equal


sites = gpd.GeoDataFrame({
    'site_id': [1, 2, 3],
    'apn': ['1', '2', '3'],
    'apn_raw': ['1', '2', '3'],
    'locapn': ['1', '2', '3'],
    'locapn_raw': ['1', '2', '3'],
    'geometry': [Point(1, 1), Point(1, 2), Point(1, 3)]
}, crs='EPSG:3310').to_crs('EPSG:3857')

permits = gpd.GeoDataFrame({
    'permit_id': [1],
    'geometry': [Point(8.5, 2)],
    'permyear': ['2017'],
    'apn': ['A'],
    'apn_raw': ['A'],
}, crs='EPSG:3310').to_crs('EPSG:3857')

class TestUtils(unittest.TestCase):
    def test_merge_on_address_lax(self):
        # Should be able to merge points within 8 meters, even if the inputs are not in a meters projection
        merged = utils.merge_on_address(sites, permits, buffer='25ft')

        expected = pd.DataFrame({
            'sites_index': [1],
            'permits_index': [0],
        })

        assert_frame_equal(merged, expected)

    def test_merge_on_address_lax_too_far(self):
        sites = gpd.GeoDataFrame({
            'site_id': [1, 2, 3],
            'geometry': [Point(1, 1), Point(1, 2), Point(1, 3)]
        }, crs='EPSG:3310').to_crs('EPSG:3857')

        # This point is more than 8 meters away
        permits = gpd.GeoDataFrame({
            'permit_id': [1],
            'geometry': [Point(10.5, 2)],
        }, crs='EPSG:3310').to_crs('EPSG:3857')

        merged = utils.merge_on_address(sites, permits, buffer='25ft')

        self.assertEqual(len(merged), 0)

    def test_calculate_pdev_for_inventory_geo(self):
        matches = utils.get_all_matches(sites, permits)
        num_matches, num_sites, match_rate = utils.calculate_pdev_for_inventory(
            sites, matches, matching_logic=utils.MatchingLogic(match_by='geo', geo_matching_buffer='5ft')
        )

        self.assertEqual(num_matches, 0)
        self.assertEqual(num_sites, 3)
        self.assertEqual(match_rate, 0)

    def test_calculate_pdev_for_inventory_geo_lax(self):
        matches = utils.get_all_matches(sites, permits)
        num_matches, num_sites, match_rate = utils.calculate_pdev_for_inventory(
            sites, matches, matching_logic=utils.MatchingLogic(match_by='geo', geo_matching_buffer='25ft')
        )
        self.assertEqual(num_matches, 1)
        self.assertEqual(num_sites, 3)
        self.assertEqual(match_rate, 1/3)

    def test_adj_pdev(self):
        self.assertEqual(utils.adj_pdev(1), 1)
        self.assertEqual(utils.adj_pdev(0), 0)
        self.assertEqual(utils.adj_pdev(0.5), 8/5*.5)


if __name__ == '__main__':
    unittest.main()
