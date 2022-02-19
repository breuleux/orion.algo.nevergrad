"""Perform integration tests for `orion.algo.nevergrad`."""
import nevergrad as ng
import pytest
from orion.benchmark.task.branin import Branin
from orion.core.utils import backward
from orion.testing.algo import BaseAlgoTests, phase

xfail = pytest.mark.xfail

_deterministic_first_point = {
    "test_seed_rng_init": xfail(reason="First generated point is deterministic")
}

_not_serializable = {
    "test_has_observed_statedict": xfail(reason="Algorithm is not serializable"),
    "test_state_dict": xfail(reason="Algorithm is not serializable"),
}

_sequential = {
    "test_seed_rng": xfail(reason="Cannot ask before tell of the last ask"),
    "test_seed_rng_init": xfail(reason="First generated point is deterministic"),
}

_no_tell_without_ask = {
    "test_observe": xfail(reason="Cannot observe a point that was not suggested"),
    "test_is_done_cardinality": xfail(
        reason="Cannot observe a point that was not suggested"
    ),
}

_not_parallel = {"test_suggest_n": xfail(reason="Cannot suggest more than one point")}

_CAN_suggest_more_than_one = {"test_suggest_n": None}

_max_trials_hangs = {"test_is_done_max_trials": "skip"}


WORKING = {
    "cGA": {},
    "ASCMADEthird": {},
    "AdaptiveDiscreteOnePlusOne": _deterministic_first_point,
    "AlmostRotationInvariantDE": {},
    "AvgMetaRecenteringNoHull": _deterministic_first_point,
    "CM": {},
    "CMA": {},
    "CauchyLHSSearch": {},
    "CauchyOnePlusOne": _deterministic_first_point,
    "CauchyScrHammersleySearch": _deterministic_first_point,
    "ChainMetaModelSQP": _CAN_suggest_more_than_one,
    "CmaFmin2": _sequential | _not_serializable,
    "DE": {},
    "DiagonalCMA": {},
    "DiscreteBSOOnePlusOne": _deterministic_first_point,
    "DiscreteLenglerOnePlusOne": _deterministic_first_point,
    "DiscreteOnePlusOne": _deterministic_first_point,
    "EDA": _no_tell_without_ask,
    "ES": {},
    "FCMA": {},
    "GeneticDE": {},
    "HaltonSearch": _deterministic_first_point,
    "HaltonSearchPlusMiddlePoint": _deterministic_first_point,
    "HammersleySearch": _deterministic_first_point,
    "HammersleySearchPlusMiddlePoint": _deterministic_first_point,
    "HullAvgMetaRecentering": _deterministic_first_point,
    "HullAvgMetaTuneRecentering": _deterministic_first_point,
    "LargeHaltonSearch": _deterministic_first_point,
    "LHSSearch": {},
    "LhsDE": {},
    "MetaModel": {},
    "MetaModelOnePlusOne": _deterministic_first_point,
    "MetaRecentering": _deterministic_first_point,
    "MetaTuneRecentering": _deterministic_first_point,
    "MixES": {},
    "MultiCMA": {},
    "MultiScaleCMA": {},
    "MutDE": {},
    "NaiveIsoEMNA": _no_tell_without_ask,
    "NaiveTBPSA": {},
    "NGO": {},
    "NGOpt": {},
    "NGOpt10": {},
    "NGOpt12": {},
    "NGOpt13": {},
    "NGOpt14": {},
    "NGOpt15": {},
    "NGOpt16": {},
    "NGOpt21": {},
    "NGOpt36": {},
    "NGOpt38": {},
    "NGOpt39": {},
    "NGOpt4": {},
    "NGOpt8": {},
    "NGOptBase": {},
    "NoisyBandit": {},
    "NoisyDE": {},
    "NoisyOnePlusOne": _deterministic_first_point,
    "NonNSGAIIES": {},
    "OnePlusOne": _deterministic_first_point,
    "OptimisticDiscreteOnePlusOne": _deterministic_first_point,
    "OptimisticNoisyOnePlusOne": _deterministic_first_point,
    "ORandomSearch": {},
    "OScrHammersleySearch": _deterministic_first_point,
    "PolyCMA": {},
    "PortfolioDiscreteOnePlusOne": _deterministic_first_point,
    "Powell": _not_parallel | _sequential,
    "PSO": {},
    "PymooNSGA2": _no_tell_without_ask | _sequential | _not_parallel,
    "QORandomSearch": {},
    "QOScrHammersleySearch": _deterministic_first_point,
    "QrDE": _deterministic_first_point,
    "RandomSearch": {},
    "RandomSearchPlusMiddlePoint": _deterministic_first_point,
    "RealSpacePSO": {},
    "RecES": {},
    "RecombiningPortfolioDiscreteOnePlusOne": _deterministic_first_point,
    "RecMixES": {},
    "RecMutDE": {},
    "RescaledCMA": {},
    "RotatedTwoPointsDE": {},
    "RotationInvariantDE": {},
    "RPowell": _deterministic_first_point | _sequential | _not_serializable,
    "RSQP": _deterministic_first_point | _sequential | _not_serializable,
    "ScrHaltonSearch": {},
    "ScrHaltonSearchPlusMiddlePoint": _deterministic_first_point,
    "ScrHammersleySearch": _deterministic_first_point,
    "ScrHammersleySearchPlusMiddlePoint": _deterministic_first_point,
    "Shiwa": {},
    "SparseDoubleFastGADiscreteOnePlusOne": _deterministic_first_point,
    "TBPSA": {},
    "TripleCMA": {},
    "TwoPointsDE": {},
}

HANGING_IN_MAX_TRIALS = {
    "ChainCMAPowell": _max_trials_hangs,
    "ChainDiagonalCMAPowell": _max_trials_hangs,
    "ChainMetaModelPowell": _max_trials_hangs,
    "ChainNaiveTBPSACMAPowell": _max_trials_hangs | _CAN_suggest_more_than_one,
    "ChainNaiveTBPSAPowell": _max_trials_hangs | _CAN_suggest_more_than_one,
}

BRANIN_FAILURES = {
    "AnisotropicAdaptiveDiscreteOnePlusOne": _deterministic_first_point | _not_parallel,
    "CMandAS2": {},
    "CMandAS3": {},
    "DoubleFastGADiscreteOnePlusOne": _deterministic_first_point,
    "MultiDiscrete": _deterministic_first_point,
    "NoisyDiscreteOnePlusOne": _deterministic_first_point,
    "ParaPortfolio": {},
    "Portfolio": {},
    "RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne": _deterministic_first_point,
    "SQPCMA": {},
}

NOT_WORKING = {
    "BO": {},
    "BOSplit": {},
    "BayesOptimBO": {},
    "DiscreteDoerrOnePlusOne": _deterministic_first_point | _not_parallel,
    "NelderMead": _sequential | _not_serializable,
    "PCABO": {},
    "SPSA": {},
    "SQP": {},
}

HANGING = {
    "Cobyla": {},
    "RCobyla": {},
}

WIP = {}

MODEL_NAMES = WORKING
# MODEL_NAMES = WIP


@pytest.fixture(autouse=True, params=MODEL_NAMES.keys())
def _config(request):
    """ Fixture that parametrizes the configuration used in the tests below. """
    tweaks = MODEL_NAMES[request.param]

    if ng.optimizers.registry[request.param].no_parallelization:
        num_workers = 1
        tweaks = {**_not_parallel, **tweaks}
    else:
        num_workers = 10

    TestNevergradOptimizer.config["model_name"] = request.param
    TestNevergradOptimizer.config["num_workers"] = num_workers

    test_name, _ = request.node.name.split("[")
    mark = tweaks.get(test_name, None)
    if mark == "skip":
        pytest.skip("Skip test")
    elif mark:
        request.node.add_marker(mark)
    yield


# Test suite for algorithms. You may reimplement some of the tests to adapt them to your algorithm
# Full documentation is available at https://orion.readthedocs.io/en/stable/code/testing/algo.html
# Look for algorithms tests in https://github.com/Epistimio/orion/blob/master/tests/unittests/algo
# for examples of customized tests.
class TestNevergradOptimizer(BaseAlgoTests):
    """Test suite for algorithm NevergradOptimizer"""

    algo_name = "nevergradoptimizer"
    config = {
        "seed": 1234,  # Because this is so random
        "budget": 200,
    }

    @phase
    def test_normal_data(self, mocker, num, attr):
        """Test that algorithm supports normal dimensions"""
        self.assert_dim_type_supported(mocker, num, attr, {"x": "normal(2, 5)"})

    def get_num(self, num):
        return min(num, 5)

    def test_optimize_branin(self):
        """Test that algorithm optimizes somehow (this is on-par with random search)"""
        MAX_TRIALS = 20  # pylint: disable=invalid-name
        task = Branin()
        space = self.create_space(task.get_search_space())
        algo = self.create_algo(config={}, space=space)
        algo.algorithm.max_trials = MAX_TRIALS
        safe_guard = 0
        trials = []
        objectives = []
        while trials or not algo.is_done:
            if safe_guard >= MAX_TRIALS:
                break

            if not trials:
                remaining = MAX_TRIALS - len(objectives)
                trials = algo.suggest(self.get_num(remaining))
                if not trials:
                    assert algo.is_done
                    break

            trial = trials.pop(0)
            results = task(trial.params["x"])
            objectives.append(results[0]["value"])
            backward.algo_observe(algo, [trial], [dict(objective=objectives[-1])])
            safe_guard += 1

        assert algo.is_done
        assert min(objectives) <= 10


# You may add other phases for test.
# See https://github.com/Epistimio/orion.algo.skopt/blob/master/tests/integration_test.py
# for an example where two phases are registered, one for the initial random step, and
# another for the optimization step with a Gaussian Process.
TestNevergradOptimizer.set_phases([("random", 0, "space.sample")])
