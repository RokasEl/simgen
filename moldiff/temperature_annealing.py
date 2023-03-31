from abc import ABC, abstractmethod

import torch


def to_float(x):
    if isinstance(x, torch.Tensor):
        return x.item()
    else:
        return x


class BaseThermostat(ABC):
    """
    Base class for thermostats.
    Thermostats map time in the generation process to a temperature.
    """

    @abstractmethod
    def __call__(self, sigma):
        """Returns 1/T at a given sigma."""
        raise NotImplementedError


class ExponentialThermostat(BaseThermostat):
    """
    Thermostat that follows an exponential decay.
    """

    def __init__(self, initial_T_log_10, final_T_log_10, sigma_max):
        self.initial_temperature_log_10 = initial_T_log_10
        self.final_temperature_log_10 = final_T_log_10
        self.sigma_max = to_float(sigma_max)

    def __call__(self, sigma):
        """Returns 1/T at a given sigma."""
        sigma = to_float(sigma)
        log_10_T = (
            self.initial_temperature_log_10 - self.final_temperature_log_10
        ) / self.sigma_max * sigma + self.final_temperature_log_10
        log_beta = -1 * log_10_T
        beta = 10**log_beta
        return beta
