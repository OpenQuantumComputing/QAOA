import numpy as np


class Statistic:
    """
    Class for collecting statistics on samples, including expectation value, variance,
    maximum, minimum, and Conditional Value at Risk (CVaR).

    See: https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf

    Attributes:
        cvar (float): Conditional Value at Risk threshold, default is 1.
        W (float): Total weight of samples.
        maxval (float): Maximum value observed.
        minval (float): Minimum value observed.
        minSols (list): List of strings corresponding to minimum values.
        maxSols (list): List of strings corresponding to maximum values.
        E (float): Expectation value of the samples.
        S (float): Variance of the samples.
        all_values (np.ndarray): Array to store all sample values for CVaR calculation.

    Methods:
        reset(): Resets all statistics to initial values.
        add_sample(value, weight, string): Adds a sample value with its weight and associated string.
        get_E(): Returns the expectation value.
        get_Variance(): Returns the variance of the samples.
        get_max(): Returns the maximum value observed.
        get_min(): Returns the minimum value observed.
        get_max_sols(): Returns the list of strings corresponding to maximum values.
        get_min_sols(): Returns the list of strings corresponding to minimum values.
        get_CVaR(): Returns the Conditional Value at Risk based on the samples.
    """

    def __init__(self, cvar=1):
        """
        Initializes the Statistic class with a specified CVaR threshold.

        Args:
            cvar (int, optional): CVaR threshold. Defaults to 1.
        """
        self.cvar = cvar
        self.reset()

    def reset(self):
        """
        Resets all statistics to their initial values.
        """
        self.W = 0
        self.maxval = float("-inf")
        self.minval = float("inf")
        self.minSols = []
        self.maxSols = []
        self.E = 0
        self.S = 0
        self.all_values = np.array([])

    def add_sample(self, value, weight, string):
        """
        Adds a sample value with its weight and associated string to the statistics.

        Args:
            value (float): The value of the sample.
            weight (float): The weight of the sample.
            string (str): The string associated with the sample.
        """
        self.W += weight
        tmp_E = self.E
        if value >= self.maxval:
            if value == self.maxval:
                self.maxSols.append(string)
            else:
                self.maxval = value
                self.maxSols = [string]
        if value <= self.minval:
            if value == self.minval:
                self.minSols.append(string)
            else:
                self.minval = value
                self.minSols = [string]

        self.maxval = max(value, self.maxval)
        self.minval = min(value, self.minval)
        self.E += weight / self.W * (value - self.E)
        self.S += weight * (value - tmp_E) * (value - self.E)
        if self.cvar < 1:
            idx = np.searchsorted(self.all_values, value)
            self.all_values = np.insert(
                self.all_values, idx, np.ones(int(weight)) * value
            )

    def get_E(self):
        """
        Returns:
            float: The expectation value of the samples.
        """
        return self.E

    def get_Variance(self):
        """
        Returns:
            float: The variance of the samples.
        """
        return self.S / (self.W - 1)

    def get_max(self):
        """
        Returns:
            float: The maximum value observed in the samples."""
        return self.maxval

    def get_min(self):
        """
        Returns:
            float: The minimum value observed in the samples.
        """
        return self.minval

    def get_max_sols(self):
        """
        Returns:
            list: The list of strings corresponding to the maximum values observed.
        """
        return self.maxSols

    def get_min_sols(self):
        """
        Returns:
            list: The list of strings corresponding to the minimum values observed.
        """
        return self.minSols

    def get_CVaR(self):
        """
        Returns:
            float: The CVaR based on the samples.
        """
        if self.cvar < 1:
            cvarK = int(np.round(self.cvar * len(self.all_values)))
            cvar = np.sum(self.all_values[-cvarK:]) / cvarK
            return cvar
        else:
            return self.get_E()
