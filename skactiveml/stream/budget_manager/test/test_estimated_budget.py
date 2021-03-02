import unittest
import numpy as np

from .._estimated_budget import (
    FixedUncertaintyBudget,
    VarUncertaintyBudget,
    SplitBudget,
)


class TestEstimatedBudget(unittest.TestCase):
    def setUp(self):
        # initialise var for sampled var tests
        self.utilities = np.array([True, False])

    def test_estimated_budget_FixedUncertaintyBudget(self):
        # init param test
        self._test_init_param_budget(FixedUncertaintyBudget)
        self._test_init_param_w(FixedUncertaintyBudget)
        self._test_init_param_num_classes(FixedUncertaintyBudget)

        # sampled param test
        self._test_sampled_param_utilities(FixedUncertaintyBudget)

        # functinality test
        # self._test_sampled_utilities(FixedUncertaintyBudget)

    def test_estimated_budget_VarUncertaintyBudget(self):
        # init param test
        self._test_init_param_budget(VarUncertaintyBudget)
        self._test_init_param_w(VarUncertaintyBudget)
        self._test_init_param_theta(VarUncertaintyBudget)
        self._test_init_param_s(VarUncertaintyBudget)

        # sampled param test
        self._test_sampled_param_utilities(VarUncertaintyBudget)

        # functinality test
        # self._test_sampled_utilities(VarUncertaintyBudget)

    def test_estimated_budget_SplitBudget(self):
        # init param test
        self._test_init_param_budget(SplitBudget)
        self._test_init_param_w(SplitBudget)
        self._test_init_param_theta(SplitBudget)
        self._test_init_param_s(SplitBudget)
        self._test_init_param_v(SplitBudget)

        # sampled param test
        self._test_sampled_param_utilities(SplitBudget)

        # functinality test
        # self._test_sampled_utilities(SplitBudget)

    def _test_init_param_budget(self, budget_manager_name):
        # budget must be defined as a float with a range of: 0 < budget <= 1
        budget_manager = budget_manager_name(budget="string")
        self.assertRaises(TypeError, budget_manager.sample, self.utilities)
        budget_manager = budget_manager_name(budget=1.1)
        self.assertRaises(ValueError, budget_manager.sample, self.utilities)
        budget_manager = budget_manager_name(budget=-1.0)
        self.assertRaises(ValueError, budget_manager.sample, self.utilities)

    def _test_init_param_w(self, budget_manager_name):
        # w must be defined as an int with a range of w > 0
        budget_manager = budget_manager_name(w="string")
        self.assertRaises(TypeError, budget_manager.sample, self.utilities)
        budget_manager = budget_manager_name(w=None)
        self.assertRaises(TypeError, budget_manager.sample, self.utilities)
        budget_manager = budget_manager_name(w=1.1)
        self.assertRaises(TypeError, budget_manager.sample, self.utilities)
        budget_manager = budget_manager_name(w=0)
        self.assertRaises(ValueError, budget_manager.sample, self.utilities)
        budget_manager = budget_manager_name(w=-1)
        self.assertRaises(ValueError, budget_manager.sample, self.utilities)

    def _test_init_param_num_classes(self, budget_manager_name):
        # num_classes must be defined as an int and greater than 0
        budget_manager = budget_manager_name(num_classes="string")
        self.assertRaises(TypeError, budget_manager.sample, self.utilities)
        budget_manager = budget_manager_name(num_classes=-1)
        self.assertRaises(ValueError, budget_manager.sample, self.utilities)
        budget_manager = budget_manager_name(num_classes=0)
        self.assertRaises(ValueError, budget_manager.sample, self.utilities)

    def _test_init_param_theta(self, budget_manager_name):
        # theta must be defined as a float
        budget_manager = budget_manager_name(theta="string")
        self.assertRaises(TypeError, budget_manager.sample, self.utilities)

    def _test_init_param_s(self, budget_manager_name):
        # s must be defined as a float with a range of: 0 < s <= 1
        budget_manager = budget_manager_name(s="string")
        self.assertRaises(TypeError, budget_manager.sample, self.utilities)
        budget_manager = budget_manager_name(s=1.1)
        self.assertRaises(ValueError, budget_manager.sample, self.utilities)
        budget_manager = budget_manager_name(s=0.0)
        self.assertRaises(ValueError, budget_manager.sample, self.utilities)
        budget_manager = budget_manager_name(s=-1.0)
        self.assertRaises(ValueError, budget_manager.sample, self.utilities)

    def _test_init_param_v(self, budget_manager_name):
        # v must be defined as an float with a range of: 0 < v < 1
        budget_manager = budget_manager_name(v="string")
        self.assertRaises(TypeError, budget_manager.sample, self.utilities)
        budget_manager = budget_manager_name(v=1.1)
        self.assertRaises(ValueError, budget_manager.sample, self.utilities)
        budget_manager = budget_manager_name(v=0.0)
        self.assertRaises(ValueError, budget_manager.sample, self.utilities)
        budget_manager = budget_manager_name(v=-1.0)
        self.assertRaises(ValueError, budget_manager.sample, self.utilities)

    def _test_sampled_param_utilities(self, budget_manager_name):
        # s must be defined as a float ndarray
        budget_manager = budget_manager_name()
        self.assertRaises(TypeError, budget_manager.sample, utilities="string")
        self.assertRaises(TypeError, budget_manager.sample, utilities=None)
        self.assertRaises(TypeError, budget_manager.sample, utilities=[10, 10])

#     def _test_sampled_utilities(self, budget_manager_name):
#         # only budget% +/- theta of utilities can be purchased

#         # initialise var for a test run to count bought samples
#         random_state = np.random.RandomState(0)
#         init_train_length = 10
#         stream_length = 10000
#         training_size = 1000

#         X, y = sklearn.datasets.make_classification(
#               n_samples=init_train_length + stream_length, 
#               random_state=, shuffle=True
#         )
#         X_init = X[:init_train_length, :]
#         y_init = y[:init_train_length]
#         X_stream = X[init_train_length:, :]
#         y_stream = y[init_train_length:]
#         clf = PWC()
#         query_strategy = FixedUncertainty(
#               clf=clf, random_state=get_randomseed(random_state),
#               budget_manager=budget_manager_name()
#         )

#         # vars to count bought samples
#         max_utilities = stream_length * 0.1
#         theta = 0.2
#         count_bought = 0
#         # run FixedUncertainty to count utilities purchased

#         X_train = deque(maxlen=training_size)
#         X_train.extend(X_init)
#         y_train = deque(maxlen=training_size)
#         y_train.extend(y_init)
#         self.clf.fit(X_train, y_train)
#         for t, (x_t, y_t) in enumerate(zip(X_stream, y_stream)):
#             sampled_indices = query_strategy.query(
#                   x_t.reshape([1, -1]), X=None, y=None
#             )
#             if len(sampled_indices):
#                 X_train.append(x_t)
#                 y_train.append(y_t)
#                 clf.fit(X_train, y_train)
#                 count_bought += 1

#         self.assertTrue(count_bought * (1 + theta) >= max_utilities)
#         self.assertTrue(count_bought * (1 - theta) <= max_utilities)
