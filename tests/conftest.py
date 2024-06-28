import pytest
from pytest_allclose import report_rmses


def pytest_terminal_summary(terminalreporter):
    report_rmses(terminalreporter)


def pytest_addoption(parser):
    parser.addoption("--ntry", type=int, default=50, help="Number of tries")
    parser.addoption("--selcase", type=int, default=-1, help="Select specific case")
    parser.addoption("--printout", action="store_true", default=False, help="Print progress")


@pytest.fixture(scope="session", autouse=True)
def ntry(request):
    return request.config.getoption("--ntry")


@pytest.fixture(scope="session", autouse=True)
def selcase(request):
    return request.config.getoption("--selcase")


@pytest.fixture(scope="session", autouse=True)
def printout(request):
    return request.config.getoption("--printout")
