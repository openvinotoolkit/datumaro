import json
import types
import unittest

from datumaro.cli.commands.info import info_command
from datumaro.util.telemetry_utils import (
    send_command_exception_info, send_command_failure_info,
    send_command_success_info, send_version_info,
)

from .requirements import Requirements, mark_requirement
from .test_util import TestException

try:
    import openvino_telemetry as tm
except ImportError:
    import datumaro.util.telemetry_stub as tm

@unittest.mock.patch(f'{tm.__name__}.Telemetry.send_event')
class TestTelemetryUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.telemetry = tm.Telemetry('Datumaro')

    @mark_requirement(Requirements.DATUM_TELEMETRY)
    def test_send_version_info(self, mock_send_event):
        version = '12.3.456'

        send_version_info(self.telemetry, version)
        mock_send_event.assert_any_call('dm', 'version', '12.3.456')

    @mark_requirement(Requirements.DATUM_TELEMETRY)
    def test_send_command_success_info(self, mock_send_event):
        cli_args = types.SimpleNamespace(loglevel='20', target='project',
            project_dir='/a/b/c', command=info_command)

        send_command_success_info(self.telemetry, cli_args,
            sensitive_args=('project_dir', 'target'))
        mock_send_event.assert_any_call('dm', 'info_result', json.dumps({
            'status': 'success',
            'loglevel': '20',
            'target': 'True',
            'project_dir': 'True'
        }))

    @mark_requirement(Requirements.DATUM_TELEMETRY)
    def test_send_command_failure_info(self, mock_send_event):
        cli_args = types.SimpleNamespace(loglevel='20', target='project',
            project_dir='/a/b/c', command=info_command)

        send_command_failure_info(self.telemetry, cli_args,
            sensitive_args=('project_dir', 'target'))
        mock_send_event.assert_any_call('dm', 'info_result', json.dumps({
            'status': 'failure',
            'loglevel': '20',
            'target': 'True',
            'project_dir': 'True'
        }))

    @mark_requirement(Requirements.DATUM_TELEMETRY)
    def test_send_command_exception_info(self, mock_send_event):
        try:
            raise TestException('Test exception')
        except TestException:
            cli_args = types.SimpleNamespace(loglevel='20', target='project',
                project_dir='/a/b/c', command=info_command)

            send_command_exception_info(self.telemetry, cli_args,
                sensitive_args=('project_dir', 'target'))

        mock_send_event.assert_any_call('dm', 'info_result', json.dumps({
            'status': 'exception',
            'loglevel': '20',
            'target': 'True',
            'project_dir': 'True'
        }))
