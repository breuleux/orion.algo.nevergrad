"""
:mod:`orion.algo.nevergrad.nevergradoptimizer -- TODO
=================================================

TODO: Write long description
"""
import pickle

import nevergrad as ng
from orion.algo.base import BaseAlgorithm
from orion.core.utils.format_trials import dict_to_trial


class SpaceConverter(dict):
    def register(self, *key):
        def deco(fn):
            self[key] = fn
            return fn

        return deco

    def __call__(self, space):
        try:
            return ng.p.Instrumentation(
                **{
                    name: self[dim.type, dim.prior_name](self, dim)
                    for name, dim in space.items()
                }
            )
        except KeyError as exc:
            raise KeyError(
                f"Dimension with type and prior: {exc.args[0]} cannot be converted to nevergrad."
            )


to_ng_space = SpaceConverter()


def _intshape(shape):
    # ng.p.Array does not accept np.int64 in shapes, they have to be ints
    return tuple(int(x) for x in shape)


@to_ng_space.register("categorical", "choices")
def _(self, dim):
    assert not dim.shape
    assert len(set(dim.prior.pk)) == 1
    return ng.p.Choice(dim.interval())


@to_ng_space.register("real", "uniform")
def _(self, dim):
    lower, upper = dim.interval()
    if dim.shape:
        return ng.p.Array(lower=lower, upper=upper, shape=_intshape(dim.shape))
    else:
        return ng.p.Scalar(lower=lower, upper=upper)


@to_ng_space.register("integer", "int_uniform")
def _(self, dim):
    return self["real", "uniform"](self, dim).set_integer_casting()


@to_ng_space.register("real", "reciprocal")
def _(self, dim):
    assert not dim.shape
    lower, upper = dim.interval()
    return ng.p.Log(lower=lower, upper=upper, exponent=2)


@to_ng_space.register("integer", "int_reciprocal")
def _(self, dim):
    return self["real", "reciprocal"](self, dim).set_integer_casting()


@to_ng_space.register("real", "norm")
def _(self, dim):
    raise NotImplementedError()


@to_ng_space.register("fidelity", "None")
def _(self, dim):
    assert not dim.shape
    assert dim.prior is None
    _, upper = dim.interval()
    # No equivalent to Fidelity space, so we always use the upper value
    return upper


class NevergradOptimizer(BaseAlgorithm):
    """TODO: Class docstring

    Parameters
    ----------
    space: `orion.algo.space.Space`
        Optimisation space with priors for each dimension.
    seed: None, int or sequence of int
        Seed for the random number generator used to sample new trials.
        Default: ``None``

    """

    requires_type = None
    requires_dist = None
    requires_shape = None

    def __init__(self, space, seed=None, budget=100, num_workers=10):
        param = to_ng_space(space)
        self.algo = ng.optimizers.NGOpt(
            parametrization=param, budget=budget, num_workers=num_workers
        )
        self.algo.enable_pickling()
        self._trial_mapping = {}
        super().__init__(space, seed=seed)

    def seed_rng(self, seed):
        """Seed the state of the random number generator.

        Parameters
        ----------
        seed: int
            Integer seed for the random number generator.

        """
        self.algo.parametrization.random_state.seed(seed)

    @property
    def state_dict(self):
        """Return a state dict that can be used to reset the state of the algorithm."""
        state_dict = super().state_dict
        state_dict["algo"] = pickle.dumps(self.algo)
        return state_dict

    def set_state(self, state_dict):
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """
        super().set_state(state_dict)
        self.algo = pickle.loads(state_dict["algo"])

    def _ask(self):
        x = self.algo.ask()
        # TODO: See what x.value[0] means exactly and in what situations it
        # might be something different from ()
        assert x.value[0] == ()
        new_trial = dict_to_trial(x.value[1], self.space)
        new_trial = self.format_trial(new_trial)
        self._trial_mapping[new_trial.id] = x
        self.register(new_trial)
        return new_trial

    def _can_produce(self):
        if self.is_done:
            return False

        algo = self.algo
        is_sequential = algo.no_parallelization
        if not is_sequential and hasattr(algo, "optim"):
            is_sequential = algo.optim.no_parallelization

        if is_sequential and algo.num_ask > (algo.num_tell - algo.num_tell_not_asked):
            return False

        return True

    def suggest(self, num):
        """Suggest a `num`ber of new sets of parameters.

        TODO: document how suggest work for this algo

        Parameters
        ----------
        num: int, optional
            Number of trials to suggest. The algorithm may return less than the number of trials
            requested.

        Returns
        -------
        list of trials or None
            A list of trials representing values suggested by the algorithm. The algorithm may opt
            out if it cannot make a good suggestion at the moment (it may be waiting for other
            trials to complete), in which case it will return None.


        Notes
        -----
        New parameters must be compliant with the problem's domain `orion.algo.space.Space`.

        """
        trials = []
        while len(trials) < num and self._can_produce():
            trials.append(self._ask())
        return trials

    def observe(self, trials):
        """Observe the `trials` new state of result.

        TODO: document how observe work for this algo

        Parameters
        ----------
        trials: list of ``orion.core.worker.trial.Trial``
           Trials from a `orion.algo.space.Space`.

        """
        for trial in trials:
            if trial.id in self._trial_mapping:
                original = self._trial_mapping[trial.id]
            else:
                original = self.algo.parametrization.spawn_child(((), trial.params))
            self.algo.tell(original, trial.objective.value)

        super().observe(trials)

    @property
    def is_done(self):
        """Return True, if an algorithm holds that there can be no further improvement."""
        # NOTE: Drop if base implementation is fine.
        return super(NevergradOptimizer, self).is_done
