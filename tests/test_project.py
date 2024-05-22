import unittest
import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'popgen'))

from project import Project

class TestPopGen(unittest.TestCase):
    def setUp(self):

        self.original_working_directory = os.getcwd()

        os.chdir(os.path.join(os.path.dirname(__file__), '..', 'data'))

    def tearDown(self):

        os.chdir(self.original_working_directory)

    def test_load_project(self):
        config_path = 'configuration_arizona.yaml'
        project = Project(config_path)
        project.load_project()
        self.assertIsNotNone(project._config)

if __name__ == '__main__':
    unittest.main()
