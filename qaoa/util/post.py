import statistics as stat
import numpy as np


def post_processing(instance, samples, K=5):
    """Performs classical post-processing on bitstrings by applying random bit flips.
    Resets and updates `self.stat`.

    Args:
        Instance (object): Object with the following attributes
            - problem (*Problem*)
            - flipper (*BitFlip*)
            - stat (*Statistic*)
        samples (dict or list or str): The bitstring(s) to be processed.
        K (int): The number of times to iterate through each bitstring and apply random bit flips.

    Returns:
        dict: A dictionary with the altered bitstrings as keys and their counts as values.
        If no better bitstring is found, the original bitstring is the key.
    """
    instance.stat.reset()
    hist_post = {}

    if isinstance(samples, str):
        samples = [samples]

    for string in samples:
        boosted = instance.flipper.boost_samples(
            problem=instance.problem, string=string, K=K
        )
        try:
            count = samples[string]
        except:
            count = 1

        instance.stat.add_sample(
            instance.problem.cost(boosted[::-1]), count, boosted[::-1]
        )
        hist_post[boosted] = hist_post.get(boosted, 0) + count
    return hist_post


def post_process_all_depths(instance, K=5):
    """Performs post-processing of `job.result().get_counts()` 100 times after each layer.

    Args:
        instance (object): Object with the following attributes
            - samplecount_hists (*dict*): A dictionary where keys are layer depths and values are histograms of sample counts.
            - stat (*Statistic*)

    Returns:
        tuple: A tuple containing
            - *np.ndarray*: Means of expectation values after post-processing for each layer.
            - *np.ndarray*: Variances of expectation values after post-processing for each layer.
    """
    exp_in_layers = {}
    exp = []
    var = []
    for d, hist in instance.samplecount_hists.items():
        if not isinstance(hist, dict):
            raise TypeError
        for i in range(100):
            post_processing(
                instance=instance,
                samples=hist,
                K=K,
            )
            exp_in_layers[d] = exp_in_layers.get(d, []) + [-instance.stat.get_CVaR()]
        exp.append(stat.mean(exp_in_layers[d]))
        var.append(stat.variance(exp_in_layers[d]))
    return (np.array(exp), np.array(var))
