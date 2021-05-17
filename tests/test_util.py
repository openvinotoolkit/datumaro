import os
import os.path as osp

from unittest import TestCase

from datumaro.util import Rollback, error_rollback
from datumaro.util.test_utils import TempTestDir
from datumaro.util.os_util import walk

import pytest
from tests.constants.requirements import Requirements
from tests.constants.datumaro_components import DatumaroComponent


@pytest.mark.components(DatumaroComponent.Datumaro)
@pytest.mark.api_other
class TestRollback(TestCase):
    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_does_not_call_on_no_error(self):
        success = True
        def cb():
            nonlocal success
            success = False

        with Rollback() as on_error:
            on_error.do(cb)

        self.assertTrue(success)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_calls_on_error(self):
        success = False
        def cb():
            nonlocal success
            success = True

        try:
            with Rollback() as on_error:
                on_error.do(cb)
                raise Exception('err')
        except Exception:
            pass
        finally:
            self.assertTrue(success)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_decorator_calls_on_error(self):
        success = False
        def cb():
            nonlocal success
            success = True

        @error_rollback('on_error')
        def foo(on_error=None):
            on_error.do(cb)
            raise Exception('err')

        try:
            foo()
        except Exception:
            pass
        finally:
            self.assertTrue(success)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_decorator_does_not_call_on_no_error(self):
        success = True
        def cb():
            nonlocal success
            success = False

        @error_rollback('on_error')
        def foo(on_error=None):
            on_error.do(cb)

        foo()

        self.assertTrue(success)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_decorator_supports_implicit_arg(self):
        success = False
        def cb():
            nonlocal success
            success = True

        @error_rollback('on_error', implicit=True)
        def foo():
            on_error.do(cb)  # noqa: F821
            raise Exception('err')

        try:
            foo()
        except Exception:
            pass
        finally:
            self.assertTrue(success)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_can_fowrard_args(self):
        success1 = False
        def cb1(a1, a2=None, ignore_errors=None):
            nonlocal success1
            if a1 == 5 and a2 == 2 and ignore_errors == None:
                success1 = True

        success2 = False
        def cb2(a1, a2=None, ignore_errors=None):
            nonlocal success2
            if a1 == 5 and a2 == 2 and ignore_errors == 4:
                success2 = True

        try:
            with Rollback() as on_error:
                on_error.do(cb1, 5, a2=2, ignore_errors=True)
                on_error.do(cb2, 5, a2=2, ignore_errors=True,
                    fwd_kwargs={'ignore_errors': 4})
                raise Exception('err')
        except Exception:
            pass
        finally:
            self.assertTrue(success1)
            self.assertTrue(success2)

@pytest.mark.components(DatumaroComponent.Datumaro)
@pytest.mark.api_other
class TestOsUtils(TestCase):
    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_can_walk_with_maxdepth(self):
        with TempTestDir() as rootdir:
            os.makedirs(osp.join(rootdir, '1', '2', '3', '4'))

            visited = set(d for d, _, _ in walk(rootdir, max_depth=2))
            self.assertEqual({
                osp.join(rootdir),
                osp.join(rootdir, '1'),
                osp.join(rootdir, '1', '2'),
            }, visited)