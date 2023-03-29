import pytest
from temperature_annealing import ExponentialThermostat


def test_expontial_thermostat():
    thermostat = ExponentialThermostat(initial_T_log_10=1, sigma_max=10)
    assert thermostat(0) == 1
    assert thermostat(10) == 0.1


def test_expontial_thermostat_with_torch():
    import torch

    thermostat = ExponentialThermostat(
        initial_T_log_10=torch.tensor(1), sigma_max=torch.tensor(10)
    )
    assert thermostat(0) == 1
    assert thermostat(10) == 0.1
