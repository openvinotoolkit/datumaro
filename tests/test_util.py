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
            on_error.do(cb) # noqa
            raise Exception('err')

        try:
            foo()
        except Exception:
            pass
        finally:
            self.assertTrue(success)

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
