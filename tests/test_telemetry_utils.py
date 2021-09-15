import unittest
import types
import json
from unittest.mock import Mock

from datumaro.cli.commands.info import info_command
from datumaro.util.telemetry_utils import (
    send_command_exception_info, send_command_failure_info,
    send_command_success_info, send_version_info,
)

try:
    import openvino_telemetry as tm
except ImportError:
    from datumaro.util import telemetry_stub as tm

class TestTelemetryUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tm.Telemetry.send_event = Mock()
        cls.telemetry = tm.Telemetry('Datumaro')

    def test_send_version_info(self):
        version = '12.3.456'

        send_version_info(self.telemetry, version)
        self.telemetry.send_event.assert_any_call('dm', 'version', '12.3.456')

    def test_send_command_success_info(self):
        cli_args = types.SimpleNamespace(loglevel='20', target='project', all='False',
            project_dir='/a/b/c', command=info_command)

        send_command_success_info(self.telemetry, cli_args, ('project_dir', 'target'))
        self.telemetry.send_event.assert_any_call('dm', 'info_result', json.dumps({
            'status': 'success',
            'loglevel': '20',
            'target': '1',
            'all': 'False',
            'project_dir': '1'
        }))

    def test_send_command_failure_info(self):
        cli_args = types.SimpleNamespace(loglevel='20', target='project', all='False',
            project_dir='/a/b/c', command=info_command)

        send_command_failure_info(self.telemetry, cli_args, ('project_dir', 'target'))
        self.telemetry.send_event.assert_any_call('dm', 'info_result', json.dumps({
            'status': 'failure',
            'loglevel': '20',
            'target': '1',
            'all': 'False',
            'project_dir': '1'
        }))

    def test_send_command_exception_info(self):
        try:
            raise ValueError('Test value exception')
        except ValueError:
            cli_args = types.SimpleNamespace(loglevel='20', target='project', all='False',
                project_dir='/a/b/c', command=info_command)

            send_command_exception_info(self.telemetry, cli_args, ('project_dir', 'target'))
            self.telemetry.send_event.assert_any_call('dm', 'info_result', json.dumps({
                'status': 'exception',
                'loglevel': '20',
                'target': '1',
                'all': 'False',
                'project_dir': '1'
            }))
