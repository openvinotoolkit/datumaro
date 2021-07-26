from unittest import TestCase
import os
import os.path as osp

from datumaro.util import Rollback, error_rollback
from datumaro.util.os_util import walk
from datumaro.util.test_utils import TestDir

from .requirements import Requirements, mark_requirement


class TestRollback(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_does_not_call_on_no_error(self):
        success = True
        def cb():
            nonlocal success
            success = False
            return 5

        with Rollback() as on_error:
            retval = on_error.do(cb)

        self.assertTrue(success)
        self.assertEqual(5, retval)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_calls_on_error(self):
        success = False
        def cb():
            nonlocal success
            success = True

        try:
            with Rollback() as on_error:
                on_error.do(cb)
                raise Exception('err')
        except Exception: # nosec - disable B110:try_except_pass check
            pass
        finally:
            self.assertTrue(success)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
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
        except Exception: # nosec - disable B110:try_except_pass check
            pass
        finally:
            self.assertTrue(success)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
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

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
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
        except Exception: # nosec - disable B110:try_except_pass check
            pass
        finally:
            self.assertTrue(success)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
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
        except Exception: # nosec - disable B110:try_except_pass check
            pass
        finally:
            self.assertTrue(success1)
            self.assertTrue(success2)

class TestOsUtils(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_walk_with_maxdepth(self):
        with TestDir() as rootdir:
            os.makedirs(osp.join(rootdir, '1', '2', '3', '4'))

            visited = set(d for d, _, _ in walk(rootdir, max_depth=2))
            self.assertEqual({
                osp.join(rootdir),
                osp.join(rootdir, '1'),
                osp.join(rootdir, '1', '2'),
            }, visited)
