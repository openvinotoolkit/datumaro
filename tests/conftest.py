#
# INTEL CONFIDENTIAL
# Copyright (c) 2021 Intel Corporation
#
# The source code contained or described herein and all documents related to
# the source code ("Material") are owned by Intel Corporation or its suppliers
# or licensors. Title to the Material remains with Intel Corporation or its
# suppliers and licensors. The Material contains trade secrets and proprietary
# and confidential information of Intel or its suppliers and licensors. The
# Material is protected by worldwide copyright and trade secret laws and treaty
# provisions. No part of the Material may be used, copied, reproduced, modified,
# published, uploaded, posted, transmitted, distributed, or disclosed in any way
# without Intel's prior express written permission.
#
# No license under any patent, copyright, trade secret or other intellectual
# property right is granted to or conferred upon you by disclosure or delivery
# of the Materials, either expressly, by implication, inducement, estoppel or
# otherwise. Any license under such intellectual property rights must be express
# and approved by Intel in writing.
#

# import e2e.fixtures
#
# from e2e.conftest_utils import *
# from e2e.utils import get_plugins_from_packages
# from tests.constants.datumaro_components import DatumaroComponent
#
# pytest_plugins = get_plugins_from_packages([e2e])
# MarksRegistry.MARK_ENUMS.extend([DatumaroComponent])


def pytest_configure(config):
    # register an additional markers
    config.addinivalue_line("markers", "api_other: mark specific test type")
    config.addinivalue_line("markers", "priority_low: mark tests with priority low")
    config.addinivalue_line("markers", "priority_medium: mark specific test type")
    config.addinivalue_line("markers", "components(component_name): mark specific component name")
    config.addinivalue_line("markers", "reqids(requirement_name): mark specific component name")
    config.addinivalue_line("markers", "bugs(bug_name): mark specific component name")

