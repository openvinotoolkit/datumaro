import numpy as np
import os.path as osp

from unittest import TestCase

from datumaro.components.project import Project
from datumaro.util.command_targets import ProjectTarget, \
    ImageTarget, SourceTarget
from datumaro.util.image import save_image
from datumaro.util.test_utils import TempTestDir

import pytest
from tests.constants.requirements import Requirements
from tests.constants.datumaro_components import DatumaroComponent


@pytest.mark.components(DatumaroComponent.Datumaro)
@pytest.mark.api_other
class CommandTargetsTest(TestCase):
    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_image_false_when_no_file(self):
        target = ImageTarget()

        status = target.test('somepath.jpg')

        self.assertFalse(status)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_image_false_when_false(self):
        with TempTestDir() as test_dir:
            path = osp.join(test_dir, 'test.jpg')
            with open(path, 'w+') as f:
                f.write('qwerty123')

            target = ImageTarget()

            status = target.test(path)

            self.assertFalse(status)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_image_true_when_true(self):
        with TempTestDir() as test_dir:
            path = osp.join(test_dir, 'test.jpg')
            save_image(path, np.ones([10, 7, 3]))

            target = ImageTarget()

            status = target.test(path)

            self.assertTrue(status)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_project_false_when_no_file(self):
        target = ProjectTarget()

        status = target.test('somepath.jpg')

        self.assertFalse(status)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_project_false_when_no_name(self):
        target = ProjectTarget(project=Project())

        status = target.test('')

        self.assertFalse(status)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_project_true_when_project_file(self):
        with TempTestDir() as test_dir:
            path = osp.join(test_dir, 'test.jpg')
            Project().save(path)

            target = ProjectTarget()

            status = target.test(path)

            self.assertTrue(status)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_project_true_when_project_name(self):
        project_name = 'qwerty'
        project = Project({
            'project_name': project_name
        })
        target = ProjectTarget(project=project)

        status = target.test(project_name)

        self.assertTrue(status)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_project_false_when_not_project_name(self):
        project_name = 'qwerty'
        project = Project({
            'project_name': project_name
        })
        target = ProjectTarget(project=project)

        status = target.test(project_name + '123')

        self.assertFalse(status)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_project_false_when_not_project_file(self):
        with TempTestDir() as test_dir:
            path = osp.join(test_dir, 'test.jpg')
            with open(path, 'w+') as f:
                f.write('wqererw')

            target = ProjectTarget()

            status = target.test(path)

            self.assertFalse(status)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_source_false_when_no_project(self):
        target = SourceTarget()

        status = target.test('qwerty123')

        self.assertFalse(status)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_source_true_when_source_exists(self):
        source_name = 'qwerty'
        project = Project()
        project.add_source(source_name)
        target = SourceTarget(project=project)

        status = target.test(source_name)

        self.assertTrue(status)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_source_false_when_source_doesnt_exist(self):
        source_name = 'qwerty'
        project = Project()
        project.add_source(source_name)
        target = SourceTarget(project=project)

        status = target.test(source_name + '123')

        self.assertFalse(status)