import unittest
from housing_elements import final_results
import geopandas as gpd
import pandas as pd

class TestFinalResults(unittest.TestCase):
    def test_analyze_realcap_input(self):
        cities = ['Berkeley', 'Albany', 'Alameda', 'Livermore', 'Fremont', 'San Ramon']
        n_sites, n_parseable, n_unlisted = final_results.analyze_realcap_input(cities)
        self.assertEqual(n_sites, 1017)
        self.assertEqual(n_parseable, 13)
        self.assertEqual(n_unlisted, 151)


if __name__ == '__main__':
    unittest.main()
