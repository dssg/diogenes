#!/usr/bin/env python
import unittest

test_modules = ['test_read',
                'test_utils',
                'test_modify',
                'test_display',
                'test_grid_search',
                'test_array_emitter']

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromNames(test_modules)
    unittest.TextTestRunner().run(suite)
