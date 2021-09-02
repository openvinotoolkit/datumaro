from contextlib import suppress
from unittest import TestCase, mock
import os
import os.path as osp

from datumaro.util import is_method_redefined
from datumaro.util.os_util import walk
from datumaro.util.scope import Scope, on_error_do, scoped, on_exit_do
from datumaro.util.test_utils import TestDir

from .requirements import Requirements, mark_requirement


class TestException(Exception):
    pass

class TestScope(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_calls_on_no_error(self):
        error_cb = mock.MagicMock()
        exit_cb = mock.MagicMock()

        with suppress(TestException), Scope() as scope:
            scope.on_error_do(error_cb)
            scope.on_exit_do(exit_cb)

        error_cb.assert_not_called()
        exit_cb.assert_called_once()

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_calls_both_stacks_on_error(self):
        error_cb = mock.MagicMock()
        exit_cb = mock.MagicMock()

        with suppress(TestException), Scope() as scope:
            scope.on_error_do(error_cb)
            scope.on_exit_do(exit_cb)
            raise TestException('err')

        error_cb.assert_called_once()
        exit_cb.assert_called_once()

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_cant_add_single_callback_in_both_stacks(self):
        cb = mock.MagicMock()

        with self.assertRaisesRegex(AssertionError, "already registered"):
            with Scope() as scope:
                scope.on_error_do(cb)
                scope.on_exit_do(cb)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_adds_cm(self):
        cm = mock.Mock()
        cm.__enter__ = mock.MagicMock(return_value=42)
        cm.__exit__ = mock.MagicMock()

        with Scope() as scope:
            retval = scope.add(cm)

        cm.__enter__.assert_called_once()
        cm.__exit__.assert_called_once()
        self.assertEqual(42, retval)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_calls_cm_on_error(self):
        cm = mock.Mock()
        cm.__enter__ = mock.MagicMock()
        cm.__exit__ = mock.MagicMock()

        with suppress(TestException), Scope() as scope:
            scope.add(cm)
            raise TestException()

        cm.__enter__.assert_called_once()
        cm.__exit__.assert_called_once()

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_decorator_calls_on_error(self):
        cb = mock.MagicMock()

        @scoped('scope')
        def foo(scope=None):
            scope.on_error_do(cb)
            raise TestException('err')

        with suppress(TestException):
            foo()

        cb.assert_called_once()

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_decorator_does_not_call_on_no_error(self):
        error_cb = mock.MagicMock()
        exit_cb = mock.MagicMock()

        @scoped('scope')
        def foo(scope=None):
            scope.on_error_do(error_cb)
            scope.on_exit_do(exit_cb)

        foo()

        error_cb.assert_not_called()
        exit_cb.assert_called_once()

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_decorator_supports_implicit_form(self):
        error_cb = mock.MagicMock()
        exit_cb = mock.MagicMock()

        @scoped
        def foo():
            on_error_do(error_cb)
            on_exit_do(exit_cb)
            raise TestException('err')

        with suppress(TestException):
            foo()

        error_cb.assert_called_once()
        exit_cb.assert_called_once()

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_fowrard_args(self):
        cb1 = mock.MagicMock()
        cb2 = mock.MagicMock()

        with suppress(TestException), Scope() as scope:
            scope.on_error_do(cb1, 5, a2=2, ignore_errors=True)
            scope.on_error_do(cb2, 5, a2=2, ignore_errors=True,
                fwd_kwargs={'ignore_errors': 4})
            raise TestException('err')

        cb1.assert_called_once_with(5, a2=2)
        cb2.assert_called_once_with(5, a2=2, ignore_errors=4)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_decorator_can_return_on_success_in_implicit_form(self):
        @scoped
        def f():
            return 42

        retval = f()

        self.assertEqual(42, retval)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_decorator_can_return_on_success_in_explicit_form(self):
        @scoped('scope')
        def f(scope=None):
            return 42

        retval = f()

        self.assertEqual(42, retval)

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


class TestMemberRedefined(TestCase):
    class Base:
        def method(self):
            pass

    def test_can_detect_no_changes_in_derived_class(self):
        class Derived(self.Base):
            pass

        self.assertFalse(is_method_redefined('method', self.Base, Derived))

    def test_can_detect_no_changes_in_derived_instance(self):
        class Derived(self.Base):
            pass

        self.assertFalse(is_method_redefined('method', self.Base, Derived()))

    def test_can_detect_changes_in_derived_class(self):
        class Derived(self.Base):
            def method(self):
                pass

        self.assertTrue(is_method_redefined('method', self.Base, Derived))

    def test_can_detect_changes_in_derived_instance(self):
        class Derived(self.Base):
            def method(self):
                pass

        self.assertTrue(is_method_redefined('method', self.Base, Derived()))

    def test_can_detect_changes_in_patched_instance(self):
        obj = self.Base()
        with mock.patch.object(obj, 'method'):
            self.assertTrue(is_method_redefined('method', self.Base, obj))
