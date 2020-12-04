from unittest import TestCase

from datumaro.util import Rollback, error_rollback


class TestRollback(TestCase):
    def test_does_not_call_on_no_error(self):
        success = True
        def cb():
            nonlocal success
            success = False

        with Rollback() as on_error:
            on_error.do(cb)

        self.assertTrue(success)

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

    def test_decorator_supports_implicit_arg(self):
        success = False
        def cb():
            nonlocal success
            success = True

        @error_rollback('on_error', implicit=True)
        def foo():
            on_error.do(cb)
            raise Exception('err')

        try:
            foo()
        except Exception:
            pass
        finally:
            self.assertTrue(success)
