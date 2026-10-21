# Ported from Meta_FTPFN@claude-init src/ppfn/prior/bnn/{mlp,bnn_prior}.py. Changes: no pfns4hpo dependency
# (its Normalize was constructed but never applied); the ECDF is cached per settings in the data root
# and generated under its own seed. See docs/decisions/0002-mf-prior-port.md.
"""The BNN behind the multi-fidelity prior: maps a hyperparameter configuration to 23 latent outputs,
which `curves.VectorizedParameterLinker` turns into the parameters of a learning-curve mixture.

Not the same BNN as `priors/bnn.py` (the BO function prior): a different family, numpy's and torch's
global RNGs, and an ECDF over all 23 outputs that only serves the parameter linking.
"""

import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ...utils.env import repo_root, var


def cache_dir() -> Path:
    return Path(var("DATA_ROOT") or repo_root() / "data") / "priors"



class MLP(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        num_layers,
        num_hidden,
        preactivation_noise_std,
        output_noise,
        init_std,
        sparseness,
    ):
        super(MLP, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.preactivation_noise_std = preactivation_noise_std
        self.output_noise = output_noise
        self.init_std = init_std
        self.sparseness = sparseness

        activation = "tanh"

        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(num_inputs, num_hidden)]
            + [torch.nn.Linear(num_hidden, num_hidden) for _ in range(num_layers - 2)]
            + [torch.nn.Linear(num_hidden, num_outputs)]
        )

        self.reset_parameters()

        self.activation = {
            "tanh": torch.nn.Tanh(),
            "relu": torch.nn.ReLU(),
            "elu": torch.nn.ELU(),
            "identity": torch.nn.Identity(),
        }[activation]

    def reset_parameters(self, init_std=None, sparseness=None):
        init_std = init_std if init_std is not None else self.init_std
        sparseness = sparseness if sparseness is not None else self.sparseness
        for linear in self.linears:
            linear.reset_parameters()

        with torch.no_grad():
            if init_std is not None:
                for linear in self.linears:
                    linear.weight.normal_(0, init_std)
                    linear.bias.normal_(0, init_std)

            if sparseness > 0.0:
                for linear in self.linears[1:-1]:
                    linear.weight /= (1.0 - sparseness) ** (1 / 2)
                    linear.weight *= torch.bernoulli(
                        torch.ones_like(linear.weight) * (1.0 - sparseness)
                    )

    def forward(self, x):
        # No input normalization, as in the original (PFNs4HPO hpo_lc_pfn_bopfn_broken.py::MLP.forward):
        # the refactor's `self.normalizer(x)` discarded its result. docs/decisions/0002-mf-prior-port.md
        for linear in self.linears[:-1]:
            x = linear(x)
            x = x + torch.randn_like(x) * self.preactivation_noise_std
            x = torch.tanh(x)
        x = self.linears[-1](x)
        return x + torch.randn_like(x) * self.output_noise




class BNNPrior(torch.nn.Module):
    output_samples = None  # per instance now (see ensure_ecdf_loaded)
    _ecdfs: dict = {}  # (num_inputs, num_outputs, N_datasets, N_per_dataset) -> sorted samples

    _lock = threading.Lock()  # Prevents race conditions during generation

    N_datasets = 10000  # Number of datasets to sample for ECDF approximation
    N_per_dataset = 1

    @classmethod
    def ensure_ecdf_loaded(cls, num_inputs, num_outputs=23):
        """Load or generate the latent ECDF for these settings, once per process and per settings.

        Changed from the original, which kept ONE class-level ECDF in the package directory: whichever
        num_inputs came first in a process then served every other num_inputs. Here it is cached per
        settings in <data root>/priors, and generated under its own fixed seed, so it neither depends
        on nor disturbs the caller's random state."""
        key = (num_inputs, num_outputs, cls.N_datasets, cls.N_per_dataset)
        if key in cls._ecdfs:
            return cls._ecdfs[key]
        with cls._lock:
            if key not in cls._ecdfs:
                cache = cache_dir()
                cache.mkdir(parents=True, exist_ok=True)
                file = cache / f"mf_bnn_ecdf_in{num_inputs}_out{num_outputs}_n{cls.N_datasets}x{cls.N_per_dataset}.npy"
                if file.exists():
                    cls._ecdfs[key] = np.load(file)
                else:
                    print(f"Generating the MF prior's latent ECDF for {num_inputs} inputs (cached in {file})...")
                    np_state = np.random.get_state()
                    with torch.random.fork_rng():
                        np.random.seed(12345 + num_inputs)
                        torch.manual_seed(12345 + num_inputs)
                        raw = cls._generate_ecdf_samples(cls.N_datasets, cls.N_per_dataset, num_inputs, num_outputs)
                    np.random.set_state(np_state)
                    cls._ecdfs[key] = np.sort(raw.numpy())
                    np.save(file, cls._ecdfs[key])
        return cls._ecdfs[key]

    def __init__(self, num_inputs, num_outputs, nn_cls=MLP):
        super(BNNPrior, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.nn_cls = nn_cls

        self.output_samples = self.ensure_ecdf_loaded(num_inputs, num_outputs)  # read by the linker

    # def load_ecdf_cache(self):
    #     if BNNPrior.CACHE_FILE.exists():
    #         print(f"Loading BNN prior ECDF cache from {BNNPrior.CACHE_FILE}")
    #         BNNPrior.output_samples = np.load(BNNPrior.CACHE_FILE)

    #     else:
    #         print("Generating BNN prior ECDF samples...")
    #         raw_samples = self._generate_ecdf_samples(self.N_datasets, self.N_per_dataset, self.num_outputs)
    #         sorted_samples = np.sort(raw_samples.numpy())
    #         np.save(self.CACHE_FILE, sorted_samples)
    #         print(f"CDF approximation saved to {self.CACHE_FILE}")

    def sample(self):
        return BNNPrior.sample_mlp(self.num_inputs, self.num_outputs, self.nn_cls)

    @classmethod
    def sample_mlp(cls, num_inputs, num_outputs, nn_cls=MLP):

        num_layers = np.random.randint(8, 16)
        num_hidden = np.random.randint(36, 150)
        init_std = np.random.uniform(0.089, 0.193)
        sparseness = 0.145
        preactivation_noise_std = np.random.uniform(
            0.0003, 0.0014
        )  # TODO: check value for this!
        output_noise = np.random.uniform(0.0004, 0.0013)

        return nn_cls(
            num_inputs,
            num_outputs,
            num_layers,
            num_hidden,
            preactivation_noise_std,
            output_noise,
            init_std,
            sparseness,
        )

    # FIXME: use class attributes?
    @classmethod
    def _generate_ecdf_samples(
        cls, N_datasets, N_per_dataset, num_inputs, num_outputs, nn_cls=MLP
    ):
        """
        Generate and cache a ECDF on the BNN output over the BNN prior.

        This method generates samples from a Bayesian Neural Network (BNN) to approximate
        the Empirical Cumulative Distribution Function (ECDF) of its output distribution. The samples
        are collected across multiple datasets and stored globally for subsequent use.

        It is done once during init of training and serves for any subsequent BNN instantiation as y-quantile function.

        Args:
            N_datasets (int): Number of datasets to sample from.
            N_per_dataset (int): Number of samples to generate per dataset.
            num_outputs (int): Dimensionality of the BNN output space.

        Returns:
            None: The method stores the sorted output samples in the class variable
            `DatasetPrior.output_sorted` for later retrieval.

        Notes:
            - This method is called only once to initialize the CDF approximation cache.
            - A total of N_datasets times N_per_dataset samples are generated (default 1M).
            - Each sample is generated by:
                1. Sampling random uniform input vectors.
                2. Generating a new dataset via `self.new_dataset()`.
                3. Computing BNN output via `self._sample_curve_params()`.
            - The outputs are flattened and sorted to create an empirical CDF.
            - Progress is printed every 100 datasets.
            - The cached CDF is used for quantile estimation and prior sampling.
        """
        output = torch.zeros((N_datasets, N_per_dataset, num_outputs))
        inputs = torch.from_numpy(
            np.random.uniform(size=(N_datasets, N_per_dataset, num_inputs))
        ).to(torch.float32)

        with torch.no_grad():
            for i in range(N_datasets):

                mlp = cls.sample_mlp(
                    num_inputs, num_outputs, nn_cls
                )  # Sample a new BNN state
                for j in range(N_per_dataset):
                    output[i, j, :] = mlp(inputs[i, j])

        return torch.flatten(output)

    # FIXME: the Link function still has the uniform method, which is basically just looking at the quantile of the cached samples!
    # def y_quantile(self, u):
    #     """Get the quantile function value for given uniform samples u in [0,1]."""
    #         if BNNPrior.output_samples is None:
    #             self.load_ecdf_cache()

    #     n_samples = BNNPrior.output_samples.shape[0]
    #     indices = (u * (n_samples - 1)).astype(int)
    #     return BNNPrior.output_samples[indices]

    #  def uniform(self, a=0.0, b=1.0): # FIXME: during the call, we could just once apply this to all outputs and store the u_values matrix!. Then we just need to apply the respective ppfs for the respective parameters!
    # u = (b - a) * self.u_values[self.counter] + a
    # self.counter += 1
    # return u
