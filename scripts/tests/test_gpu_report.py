import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

spec = importlib.util.spec_from_file_location('gpu_runner', Path(__file__).parents[1] / 'test_gpu.py')
gpu_runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu_runner)


class TestGpuReport(unittest.TestCase):
    def test_missing_cache_dependency_stops_before_launch(self):
        with patch.object(gpu_runner.sys, 'argv', ['gpu-runner']), \
             patch.object(gpu_runner.importlib.util, 'find_spec', return_value=None), \
             patch.object(gpu_runner.os, 'execvpe') as runner, \
             patch.object(gpu_runner.subprocess, 'call') as tests:
            with self.assertRaises(SystemExit) as raised:
                gpu_runner.main()
        self.assertEqual(raised.exception.code, 2)
        runner.assert_not_called()
        tests.assert_not_called()

    def report(self, count=10, status=None, duplicate=False):
        with tempfile.TemporaryDirectory() as directory:
            root = ET.Element('testsuite')
            for i in range(count):
                case = ET.SubElement(root, 'testcase', name=f'test_{0 if duplicate else i}')
                if i == 0 and status:
                    ET.SubElement(case, status)
            report = Path(directory) / 'report.xml'
            ET.ElementTree(root).write(report)
            return gpu_runner.passed_gates(report)

    def test_complete_gpu_gates(self):
        self.assertTrue(self.report())

    def test_skip_failure_error_missing_and_duplicate_are_incomplete(self):
        for status in ('skipped', 'failure', 'error'):
            with self.subTest(status=status):
                self.assertFalse(self.report(status=status))
        self.assertFalse(self.report(count=9))
        self.assertFalse(self.report(duplicate=True))


if __name__ == '__main__':
    unittest.main()
