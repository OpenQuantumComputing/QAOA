import numpy as np
import matplotlib.pyplot as plt
import datetime

from qiskit_finance.data_providers import RandomDataProvider

class AssetData:

    def __init__(self, N_assets, num_days = 101,
                 seed = 0, start_time = datetime.datetime(2020, 1, 1)):
        """
        init function that initializes member variables

        :param params: additional parameters
        """
        self.N = N_assets
        self.num_days = num_days
        self.start_time = start_time
        self.end_time = start_time + datetime.timedelta(self.num_days)

        self.tickers = [("TICKER%s" % i) for i in range(self.N)]
        self.fin_data = RandomDataProvider(
            tickers = self.tickers,
            start=self.start_time,
            end = self.end_time,
            seed = seed
        )
        self.fin_data.run()

        self.cov_matrix = self.fin_data.get_period_return_covariance_matrix()
        self.exp_return = self.fin_data.get_period_return_mean_vector()

    def plotAssets(self, figsize=(12,4)):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 3)
        axs = [None]*2
        axs[0] = fig.add_subplot(gs[0, 0:2]) 
        t = [self.start_time + datetime.timedelta(dt) for dt in range(self.num_days)]
        for (i, ticker) in enumerate(self.tickers):
            axs[0].plot(t, self.fin_data._data[i], label=ticker)
        axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=-30)
        axs[0].legend()
        axs[0].set_title("time development")

        axs[1] = fig.add_subplot(gs[0,2])
        im = axs[1].imshow(self.cov_matrix)
        fig.colorbar(im, ax=axs[1], shrink=0.8)
        axs[1].set_title("Period return cov. matrix")

    def plotPeriodReturns(self, figsize=(8,4)):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 3)
        axs = [None]*2
        axs[0] = fig.add_subplot(gs[0, 0:2]) 
        t = [self.start_time + datetime.timedelta(dt) for dt in range(self.num_days)]
        for (i, ticker) in enumerate(self.tickers):
            axs[0].plot(t, self.fin_data._data[i], label=ticker)
        axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=-30)
        axs[0].legend()
        axs[0].set_title("time development")
