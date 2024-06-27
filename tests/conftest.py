import pytest
from pytest_allclose import report_rmses


def pytest_terminal_summary(terminalreporter):
    report_rmses(terminalreporter)


def pytest_addoption(parser):
    parser.addoption("--ntry", type=int, default=50, help="Number of tries")

def pytest_generate_tests(metafunc):
    option_value = metafunc.config.option.ntry
    if "ntry" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("ntry", [option_value])
