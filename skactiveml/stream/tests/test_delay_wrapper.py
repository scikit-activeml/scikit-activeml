import unittest
import numpy as np
from skactiveml.stream import (
    ForgettingWrapper,
    FuzzyDelaySimulationWrapper,
    BaggingDelaySimulationWrapper,
)
from skactiveml.stream import Split
from skactiveml.classifier import PWC
from abc import ABC


class QueryTests(ABC):
    def _test_init(
        self,
        delay_wrapper,
        X_cand,
        X,
        y,
        tX,
        ty,
        tX_cand,
        ty_cand,
        acquisitions,
        sample_weight,
    ):
        self.delay_wrapper = delay_wrapper
        self.X_cand = X_cand
        self.X = X
        self.y = y
        self.tX = tX
        self.ty = ty
        self.tX_cand = tX_cand
        self.ty_cand = ty_cand
        self.acquisitions = acquisitions
        self.sample_weight = sample_weight

    # query param test
    def test_query_param_X_cand(self):
        # X_cand must be defined as a two dimensinal array
        self.assertRaises(
            ValueError,
            self.delay_wrapper.query,
            X_cand=np.ones(5),
            X=self.X,
            y=self.y,
            tX=self.tX,
            ty=self.ty,
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            simulate=False,
            return_utilities=False,
        )
        self.assertRaises(
            ValueError,
            self.delay_wrapper.query,
            X_cand=None,
            X=self.X,
            y=self.y,
            tX=self.tX,
            ty=self.ty,
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            simulate=False,
            return_utilities=False,
        )
        self.assertRaises(
            ValueError,
            self.delay_wrapper.query,
            X_cand=1,
            X=self.X,
            y=self.y,
            tX=self.tX,
            ty=self.ty,
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            simulate=False,
            return_utilities=False,
        )

    def test_query_param_X(self):
        # X must be defined as a two dimensinal array and must be equal in
        # length to y
        self.assertRaises(
            ValueError,
            self.delay_wrapper.query,
            X_cand=self.X_cand,
            X=None,
            y=self.y,
            tX=self.tX,
            ty=self.ty,
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            simulate=False,
            return_utilities=False,
        )
        self.assertRaises(
            ValueError,
            self.delay_wrapper.query,
            X_cand=self.X_cand,
            X=np.ones(5),
            y=self.y,
            tX=self.tX,
            ty=self.ty,
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            simulate=False,
            return_utilities=False,
        )
        self.assertRaises(
            ValueError,
            self.delay_wrapper.query,
            X_cand=self.X_cand,
            X=self.X[1:],
            y=self.y,
            tX=self.tX,
            ty=self.ty,
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            simulate=False,
            return_utilities=False,
        )

    def test_query_param_y(self):
        # y must be defined as a one Dimensional array and must be equal in
        # length to X
        self.assertRaises(
            ValueError,
            self.delay_wrapper.query,
            X_cand=self.X_cand,
            X=self.X,
            y=np.zeros((len(self.y), 2)),
            tX=self.tX,
            ty=self.ty,
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            simulate=False,
            return_utilities=False,
        )
        self.assertRaises(
            TypeError,
            self.delay_wrapper.query,
            X_cand=self.X_cand,
            X=self.X,
            y=None,
            tX=self.tX,
            ty=self.ty,
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            simulate=False,
            return_utilities=False,
        )
        self.assertRaises(
            ValueError,
            self.delay_wrapper.query,
            X_cand=self.X_cand,
            X=self.X,
            y=self.y[1:],
            tX=self.tX,
            ty=self.ty,
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            simulate=False,
            return_utilities=False,
        )

    def test_query_param_tX(self):
        # tX needs to be a list of numeric type and same size as ty
        self.assertRaises(
            ValueError,
            self.delay_wrapper.query,
            X_cand=self.X_cand,
            X=self.X,
            y=self.y,
            tX=None,
            ty=self.ty,
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            simulate=False,
            return_utilities=False,
        )
        self.assertRaises(
            ValueError,
            self.delay_wrapper.query,
            X_cand=self.X_cand,
            X=self.X,
            y=self.y,
            tX=self.tX[1:],
            ty=self.ty,
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            simulate=False,
            return_utilities=False,
        )
        self.assertRaises(
            ValueError,
            self.delay_wrapper.query,
            X_cand=self.X_cand,
            X=self.X,
            y=self.y,
            tX=np.ones(5),
            ty=self.ty,
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            simulate=False,
            return_utilities=False,
        )

    def test_query_param_ty(self):
        # ty needs to be a list of numeric type and same size as tX
        self.assertRaises(
            ValueError,
            self.delay_wrapper.query,
            X_cand=self.X_cand,
            X=self.X,
            y=self.y,
            tX=self.tX,
            ty=self.ty[1:],
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            simulate=False,
            return_utilities=False,
        )
        self.assertRaises(
            ValueError,
            self.delay_wrapper.query,
            X_cand=self.X_cand,
            X=self.X,
            y=self.y,
            tX=self.tX,
            ty=None,
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            simulate=False,
            return_utilities=False,
        )
        self.assertRaises(
            ValueError,
            self.delay_wrapper.query,
            X_cand=self.X_cand,
            X=self.X,
            y=self.y,
            tX=self.tX,
            ty=np.ones(5),
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            simulate=False,
            return_utilities=False,
        )

    def test_query_param_tX_cand(self):
        # tX_cand needs to be a list of numeric type and same size as ty_cand
        self.assertRaises(
            ValueError,
            self.delay_wrapper.query,
            X_cand=self.X_cand,
            X=self.X,
            y=self.y,
            tX=self.tX,
            ty=self.ty,
            tX_cand=None,
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            simulate=False,
            return_utilities=False,
        )
        self.assertRaises(
            ValueError,
            self.delay_wrapper.query,
            X_cand=self.X_cand,
            X=self.X,
            y=self.y,
            tX=self.tX,
            ty=self.ty,
            tX_cand=self.tX_cand[1:],
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            simulate=False,
            return_utilities=False,
        )
        self.assertRaises(
            ValueError,
            self.delay_wrapper.query,
            X_cand=self.X_cand,
            X=self.X,
            y=self.y,
            tX=self.tX,
            ty=self.ty,
            tX_cand=np.ones(5),
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            simulate=False,
            return_utilities=False,
        )

    def test_query_param_ty_cand(self):
        # ty_cand needs to be a list of numeric type and same size as tX_cand
        self.assertRaises(
            ValueError,
            self.delay_wrapper.query,
            X_cand=self.X_cand,
            X=self.X,
            y=self.y,
            tX=self.tX,
            ty=self.ty,
            tX_cand=self.tX_cand,
            ty_cand=None,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            simulate=False,
            return_utilities=False,
        )
        self.assertRaises(
            ValueError,
            self.delay_wrapper.query,
            X_cand=self.X_cand,
            X=self.X,
            y=self.y,
            tX=self.tX,
            ty=self.ty,
            tX_cand=self.tX_cand,
            ty_cand=np.ones(5),
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            simulate=False,
            return_utilities=False,
        )
        self.assertRaises(
            ValueError,
            self.delay_wrapper.query,
            X_cand=self.X_cand,
            X=self.X,
            y=self.y,
            tX=self.tX,
            ty=self.ty,
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand[1:],
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            simulate=False,
            return_utilities=False,
        )

    def test_query_param_return_utilities(self):
        # return_utilities needs to be a boolean
        self.assertRaises(
            TypeError,
            self.delay_wrapper.query,
            X_cand=self.X_cand,
            X=self.X,
            y=self.y,
            tX=self.tX,
            ty=self.ty,
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            simulate=False,
            return_utilities="string",
        )
        self.assertRaises(
            TypeError,
            self.delay_wrapper.query,
            X_cand=self.X_cand,
            X=self.X,
            y=self.y,
            tX=self.tX,
            ty=self.ty,
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            simulate=False,
            return_utilities=1,
        )

    def test_query_param_simulate(self):
        # simulate needs to be a boolean
        self.assertRaises(
            TypeError,
            self.delay_wrapper.query,
            X_cand=self.X_cand,
            X=self.X,
            y=self.y,
            tX=self.tX,
            ty=self.ty,
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            simulate="string",
            return_utilities=False,
        )
        self.assertRaises(
            TypeError,
            self.delay_wrapper.query,
            X_cand=self.X_cand,
            X=self.X,
            y=self.y,
            tX=self.tX,
            ty=self.ty,
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            simulate=1,
            return_utilities=False,
        )

    def test_query_param_acquisitions(self):
        # acquisitions needs to be a bool list
        self.assertRaises(
            TypeError,
            self.delay_wrapper.query,
            X_cand=self.X_cand,
            X=self.X,
            y=self.y,
            tX=self.tX,
            ty=self.ty,
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions=None,
            sample_weight=self.sample_weight,
            simulate=False,
            return_utilities=False,
        )
        self.assertRaises(
            TypeError,
            self.delay_wrapper.query,
            X_cand=self.X_cand,
            X=self.X,
            y=self.y,
            tX=self.tX,
            ty=self.ty,
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions="string",
            sample_weight=self.sample_weight,
            simulate=False,
            return_utilities=False,
        )

    def test_query_param_sample_weight(self):
        # sample weight needs to be a list that can be convertet to float
        # equal in size of y
        self.assertRaises(
            TypeError,
            self.delay_wrapper.query,
            X_cand=self.X_cand,
            X=self.X,
            y=self.y,
            tX=self.tX,
            ty=self.ty,
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight="string",
            simulate=False,
            return_utilities=False,
        )
        self.assertRaises(
            TypeError,
            self.delay_wrapper.query,
            X_cand=self.X_cand,
            X=self.X,
            y=self.y,
            tX=self.tX,
            ty=self.ty,
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=["string", "numbers", "test"],
            simulate=False,
            return_utilities=False,
        )
        self.assertRaises(
            ValueError,
            self.delay_wrapper.query,
            X_cand=self.X_cand,
            X=self.X,
            y=self.y,
            tX=self.tX,
            ty=self.ty,
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=[1],
            simulate=False,
            return_utilities=False,
        )


class TestForgettingWrapper(unittest.TestCase, QueryTests):
    def setUp(self):
        self.random_state = np.random.RandomState(0)
        self.X = np.array([[0], [0], [1000]])
        self.X_cand = np.array([self.X[2]])
        self.clf = PWC(classes=[0, 1])
        self.query_strategies = Split(
            clf=self.clf, random_state=self.random_state.randint(2 ** 31 - 1)
        )
        self.delay_prior = 0.001
        self.w_train = 2.5
        self.acquisitions = np.full(3, True)
        self.sample_weight = None
        self.tX_cand = np.array([1])
        self.ty_cand = np.array([2])
        self.y = np.array([0, 1, 0])
        self.tX = np.array([0, 0, -10])
        self.ty = self.tX
        self.kwargs = dict(
            X_cand=self.X_cand,
            X=self.X,
            y=self.y,
            tX=self.tX,
            ty=self.ty,
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            return_utilities=True,
        )
        self.delay_wrapper = ForgettingWrapper(
            base_query_strategy=self.query_strategies,
            w_train=self.w_train,
            random_state=self.random_state,
        )
        self._test_init(
            delay_wrapper=self.delay_wrapper,
            X_cand=self.X_cand,
            X=self.X,
            y=self.y,
            tX=self.tX,
            ty=self.ty,
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
        )

    # def test_FO(self):
    #     # init param test
    #     # self._test_init_param_base_query_strategy()
    #     # self._test_init_param_w_train()
    #     # self._test_init_param_random_state()

    #     # query param test
    #     self.test_query_param_X_cand()
    #     self.test_query_param_X()
    #     self.test_query_param_y()
    #     self.test_query_param_tX()
    #     self.test_query_param_ty()
    #     self.test_query_param_tX_cand()
    #     self.test_query_param_ty_cand()
    #     self.test_query_param_return_utilities()
    #     self.test_query_param_simulate()
    #     self.test_query_param_acquisitions()

    #     # test functionality
    #     # self._test_query_FO()

    # query param test
    def test_query_param_X_cand(self):
        return super().test_query_param_X_cand()

    def test_query_param_tX(self):
        return super().test_query_param_tX()

    def test_query_param_tX_cand(self):
        return super().test_query_param_tX_cand()

    def test_query_param_ty(self):
        return super().test_query_param_ty()

    def test_query_param_ty_cand(self):
        return super().test_query_param_ty_cand()

    def test_query_param_acquisitions(self):
        return super().test_query_param_acquisitions()

    def test_query_param_return_utilities(self):
        return super().test_query_param_return_utilities()

    def test_query_param_sample_weight(self):
        return super().test_query_param_sample_weight()

    def test_query_param_simulate(self):
        return super().test_query_param_simulate()

    # init param test
    def test_init_param_base_query_strategy(self):
        delay_wrapper = ForgettingWrapper(
            base_query_strategy="string",
            w_train=self.w_train,
            random_state=self.random_state,
        )
        self.assertRaises(TypeError, delay_wrapper.query, **(self.kwargs))
        delay_wrapper = ForgettingWrapper(
            base_query_strategy=1,
            w_train=self.w_train,
            random_state=self.random_state,
        )
        self.assertRaises(TypeError, delay_wrapper.query, **(self.kwargs))

    def test_init_param_random_state(self):
        delay_wrapper = ForgettingWrapper(
            base_query_strategy=self.query_strategies,
            w_train=self.w_train,
            random_state="string",
        )
        self.assertRaises(ValueError, delay_wrapper.query, **(self.kwargs))

    def test_init_param_w_train(self):
        delay_wrapper = ForgettingWrapper(
            base_query_strategy=self.query_strategies,
            w_train="string",
            random_state=self.random_state,
        )
        self.assertRaises(TypeError, delay_wrapper.query, **(self.kwargs))
        delay_wrapper = ForgettingWrapper(
            base_query_strategy=self.query_strategies,
            w_train=-1.0,
            random_state=self.random_state,
        )
        self.assertRaises(ValueError, delay_wrapper.query, **(self.kwargs))

    # query param test
    def test_query_param_X(self):
        return super().test_query_param_X()

    def test_query_param_y(self):
        return super().test_query_param_y()

    # query stategy test
    def test_query_FO(self):
        y = np.array([0, 1, 0])
        tX = np.array([0, 0, -10])
        ty = tX

        _, util = self.delay_wrapper.query(
            X_cand=self.X_cand,
            X=self.X,
            y=y,
            sample_weight=self.sample_weight,
            tX=tX,
            ty=ty,
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            return_utilities=True,
        )
        self.assertEqual(0.5, util[0])

        tX = np.array([0, 0, 0])
        ty = tX

        _, util = self.delay_wrapper.query(
            X_cand=self.X_cand,
            X=self.X,
            y=y,
            sample_weight=self.sample_weight,
            tX=tX,
            ty=ty,
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            return_utilities=True,
        )
        self.assertEqual(1.0, util[0])


class TestFuzzyDelaySimulationWrapper(unittest.TestCase, QueryTests):
    def setUp(self):
        self.random_state = np.random.RandomState(0)
        self.X = np.array([[0], [0], [1000]])
        self.X_cand = np.array([self.X[2]])
        self.clf = PWC(classes=[0, 1])
        self.query_strategies = Split(
            clf=self.clf, random_state=self.random_state.randint(2 ** 31 - 1)
        )
        self.delay_prior = 0.001
        self.acquisitions = np.full(3, True)
        self.sample_weight = None
        self.tX_cand = np.array([1])
        self.ty_cand = np.array([2])
        self.y = np.array([0, 1, 0])
        self.tX = np.array([0, 0, -10])
        self.ty = self.tX
        self.kwargs = dict(
            X_cand=self.X_cand,
            X=self.X,
            y=self.y,
            tX=self.tX,
            ty=self.ty,
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            return_utilities=False,
            simulate=False,
        )
        self.delay_wrapper = FuzzyDelaySimulationWrapper(
            base_query_strategy=self.query_strategies,
            delay_prior=self.delay_prior,
            random_state=self.random_state,
            clf=self.clf,
        )
        self._test_init(
            delay_wrapper=self.delay_wrapper,
            X_cand=self.X_cand,
            X=self.X,
            y=self.y,
            tX=self.tX,
            ty=self.ty,
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
        )

    # def test_FI(self):
    #     # init param test
    #     # self._test_init_param_base_query_strategy()
    #     # self._test_init_param_delay_prior()
    #     # self._test_init_param_clf()
    #     # self._test_init_param_random_state()

    #     # query param test
    #     self.test_query_param_X_cand()
    #     self.test_query_param_X()
    #     self.test_query_param_y()
    #     self.test_query_param_tX()
    #     self.test_query_param_ty()
    #     self.test_query_param_tX_cand()
    #     self.test_query_param_ty_cand()
    #     self.test_query_param_return_utilities()
    #     self.test_query_param_simulate()
    #     self.test_query_param_sample_weight()
    #     self.test_query_param_acquisitions()

    #     # test functionality
    #     # self._test_query_FI()

    # query param test
    def test_query_param_X_cand(self):
        return super().test_query_param_X_cand()

    def test_query_param_X(self):
        return super().test_query_param_X()

    def test_query_param_y(self):
        return super().test_query_param_y()

    def test_query_param_tX(self):
        return super().test_query_param_tX()

    def test_query_param_tX_cand(self):
        return super().test_query_param_tX_cand()

    def test_query_param_ty(self):
        return super().test_query_param_ty()

    def test_query_param_ty_cand(self):
        return super().test_query_param_ty_cand()

    def test_query_param_acquisitions(self):
        return super().test_query_param_acquisitions()

    def test_query_param_return_utilities(self):
        return super().test_query_param_return_utilities()

    def test_query_param_sample_weight(self):
        return super().test_query_param_sample_weight()

    def test_query_param_simulate(self):
        return super().test_query_param_simulate()

    # init param test
    def test_init_param_base_query_strategy(self):
        delay_wrapper = FuzzyDelaySimulationWrapper(
            base_query_strategy="string",
            delay_prior=self.delay_prior,
            random_state=self.random_state,
            clf=self.clf,
        )
        self.assertRaises(TypeError, delay_wrapper.query, **(self.kwargs))
        delay_wrapper = FuzzyDelaySimulationWrapper(
            base_query_strategy=1,
            delay_prior=self.delay_prior,
            random_state=self.random_state,
            clf=self.clf,
        )
        self.assertRaises(TypeError, delay_wrapper.query, **(self.kwargs))

    def test_init_param_random_state(self):
        delay_wrapper = FuzzyDelaySimulationWrapper(
            base_query_strategy=self.query_strategies,
            delay_prior=self.delay_prior,
            clf=self.clf,
            random_state="string",
        )
        self.assertRaises(ValueError, delay_wrapper.query, **(self.kwargs))

    def test_init_param_clf(self):
        delay_wrapper = FuzzyDelaySimulationWrapper(
            base_query_strategy=self.query_strategies,
            delay_prior=self.delay_prior,
            random_state=self.random_state,
            clf="string",
        )
        self.assertRaises(TypeError, delay_wrapper.query, **(self.kwargs))
        delay_wrapper = FuzzyDelaySimulationWrapper(
            base_query_strategy=self.query_strategies,
            delay_prior=self.delay_prior,
            random_state=self.random_state,
            clf=1,
        )
        self.assertRaises(TypeError, delay_wrapper.query, **(self.kwargs))

    def test_init_param_delay_prior(self):
        delay_wrapper = FuzzyDelaySimulationWrapper(
            base_query_strategy=self.query_strategies,
            delay_prior="string",
            random_state=self.random_state,
            clf=self.clf,
        )
        self.assertRaises(TypeError, delay_wrapper.query, **(self.kwargs))
        delay_wrapper = FuzzyDelaySimulationWrapper(
            base_query_strategy=self.query_strategies,
            delay_prior=-1.0,
            random_state=self.random_state,
            clf=self.clf,
        )
        self.assertRaises(ValueError, delay_wrapper.query, **(self.kwargs))
        delay_wrapper = FuzzyDelaySimulationWrapper(
            base_query_strategy=self.query_strategies,
            delay_prior=0.0,
            random_state=self.random_state,
            clf=self.clf,
        )

    # query stategy test
    def test_query_FI(self):
        missing_label = self.clf.missing_label
        X = np.array([[0], [0], [0]])
        X_cand = np.array([X[2]])
        y = np.array([1, 1, missing_label])
        tX = np.array([0, 0, 0])
        ty = np.array([0, 0, 1.5])

        _, util = self.delay_wrapper.query(
            X_cand,
            X,
            y,
            tX,
            ty,
            self.tX_cand,
            self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            return_utilities=True,
        )
        self.assertGreaterEqual(1.0, util[0])

        tX = np.array([0, 0, 0])
        ty = np.array([0, 0, 2.5])

        _, util = self.delay_wrapper.query(
            X_cand,
            X,
            y,
            tX,
            ty,
            self.tX_cand,
            self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            return_utilities=True,
        )
        self.assertEqual(1.0, util[0])


class TestBaggingDelaySimulationWrapper(unittest.TestCase, QueryTests):
    def setUp(self):
        self.random_state = np.random.RandomState(0)
        self.X = np.array([[0], [0], [1000]])
        self.X_cand = np.array([self.X[2]])
        self.clf = PWC(classes=[0, 1])
        self.query_strategies = Split(
            clf=self.clf, random_state=self.random_state.randint(2 ** 31 - 1)
        )
        self.delay_prior = 0.001
        self.K = 2
        self.acquisitions = np.full(3, True)
        self.sample_weight = None
        self.tX_cand = np.array([1])
        self.ty_cand = np.array([2])
        self.y = np.array([0, 1, 0])
        self.tX = np.array([0, 0, -10])
        self.ty = self.tX
        self.kwargs = dict(
            X_cand=self.X_cand,
            X=self.X,
            y=self.y,
            tX=self.tX,
            ty=self.ty,
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            return_utilities=False,
            simulate=False,
        )
        self.delay_wrapper = BaggingDelaySimulationWrapper(
            base_query_strategy=self.query_strategies,
            K=self.K,
            delay_prior=self.delay_prior,
            random_state=self.random_state,
            clf=self.clf,
        )
        self._test_init(
            delay_wrapper=self.delay_wrapper,
            X_cand=self.X_cand,
            X=self.X,
            y=self.y,
            tX=self.tX,
            ty=self.ty,
            tX_cand=self.tX_cand,
            ty_cand=self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
        )

    # def test_BI(self):
    #     # init param test
    #     # self._test_init_param_base_query_strategy()
    #     # self._test_init_param_K()
    #     # self._test_init_param_delay_prior()
    #     # self._test_init_param_clf()
    #     # self._test_init_param_random_state()

    #     # query param test
    #     self.test_query_param_X_cand()
    #     self.test_query_param_X()
    #     self.test_query_param_y()
    #     self.test_query_param_tX()
    #     self.test_query_param_ty()
    #     self.test_query_param_tX_cand()
    #     self.test_query_param_ty_cand()
    #     self.test_query_param_return_utilities()
    #     self.test_query_param_simulate()
    #     self.test_query_param_sample_weight()
    #     self.test_query_param_acquisitions()

    #     # test functionality
    #     # self._test_query_BI()

    # query param test
    def test_query_param_X_cand(self):
        return super().test_query_param_X_cand()

    def test_query_param_X(self):
        return super().test_query_param_X()

    def test_query_param_y(self):
        return super().test_query_param_y()

    def test_query_param_tX(self):
        return super().test_query_param_tX()

    def test_query_param_tX_cand(self):
        return super().test_query_param_tX_cand()

    def test_query_param_ty(self):
        return super().test_query_param_ty()

    def test_query_param_ty_cand(self):
        return super().test_query_param_ty_cand()

    def test_query_param_acquisitions(self):
        return super().test_query_param_acquisitions()

    def test_query_param_return_utilities(self):
        return super().test_query_param_return_utilities()

    def test_query_param_sample_weight(self):
        return super().test_query_param_sample_weight()

    def test_query_param_simulate(self):
        return super().test_query_param_simulate()

    # init param test
    def test_init_param_base_query_strategy(self):
        delay_wrapper = BaggingDelaySimulationWrapper(
            base_query_strategy="string",
            K=self.K,
            delay_prior=self.delay_prior,
            random_state=self.random_state,
            clf=self.clf,
        )
        self.assertRaises(TypeError, delay_wrapper.query, **(self.kwargs))
        delay_wrapper = BaggingDelaySimulationWrapper(
            base_query_strategy=1,
            K=self.K,
            delay_prior=self.delay_prior,
            random_state=self.random_state,
            clf=self.clf,
        )
        self.assertRaises(TypeError, delay_wrapper.query, **(self.kwargs))

    def test_init_param_random_state(self):
        delay_wrapper = BaggingDelaySimulationWrapper(
            base_query_strategy=self.query_strategies,
            K=self.K,
            delay_prior=self.delay_prior,
            clf=self.clf,
            random_state="string",
        )
        self.assertRaises(ValueError, delay_wrapper.query, **(self.kwargs))

    def test_init_param_clf(self):
        delay_wrapper = BaggingDelaySimulationWrapper(
            base_query_strategy=self.query_strategies,
            K=self.K,
            delay_prior=self.delay_prior,
            random_state=self.random_state,
            clf="string",
        )
        self.assertRaises(TypeError, delay_wrapper.query, **(self.kwargs))
        delay_wrapper = BaggingDelaySimulationWrapper(
            base_query_strategy=self.query_strategies,
            K=self.K,
            delay_prior=self.delay_prior,
            random_state=self.random_state,
            clf=1,
        )
        self.assertRaises(TypeError, delay_wrapper.query, **(self.kwargs))

    def test_init_param_delay_prior(self):
        delay_wrapper = BaggingDelaySimulationWrapper(
            base_query_strategy=self.query_strategies,
            K=self.K,
            delay_prior="string",
            random_state=self.random_state,
            clf=self.clf,
        )
        self.assertRaises(TypeError, delay_wrapper.query, **(self.kwargs))
        delay_wrapper = BaggingDelaySimulationWrapper(
            base_query_strategy=self.query_strategies,
            K=self.K,
            delay_prior=-1.0,
            random_state=self.random_state,
            clf=self.clf,
        )
        self.assertRaises(ValueError, delay_wrapper.query, **(self.kwargs))
        delay_wrapper = BaggingDelaySimulationWrapper(
            base_query_strategy=self.query_strategies,
            K=self.K,
            delay_prior=0.0,
            random_state=self.random_state,
            clf=self.clf,
        )

    def test_init_param_K(self):
        delay_wrapper = BaggingDelaySimulationWrapper(
            base_query_strategy=self.query_strategies,
            K="string",
            delay_prior=self.delay_prior,
            random_state=self.random_state,
            clf=self.clf,
        )
        self.assertRaises(TypeError, delay_wrapper.query, **(self.kwargs))
        delay_wrapper = BaggingDelaySimulationWrapper(
            base_query_strategy=self.query_strategies,
            K=-1,
            delay_prior=self.delay_prior,
            random_state=self.random_state,
            clf=self.clf,
        )
        self.assertRaises(ValueError, delay_wrapper.query, **(self.kwargs))

    # query stategy test
    def test_query_BI(self):
        missing_label = self.clf.missing_label
        y = np.array([0, 1, missing_label])
        tX = np.array([0, 0, 0])
        ty = np.array([0, 0, 2.5])

        _, util = self.delay_wrapper.query(
            self.X_cand,
            self.X,
            y,
            tX,
            ty,
            self.tX_cand,
            self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            return_utilities=True,
        )
        self.assertEqual(0.5, util[0])

        tX = np.array([0, 0, 0])
        ty = np.array([0, 0, 1.5])

        _, util = self.delay_wrapper.query(
            self.X_cand,
            self.X,
            y,
            tX,
            ty,
            self.tX_cand,
            self.ty_cand,
            acquisitions=self.acquisitions,
            sample_weight=self.sample_weight,
            return_utilities=True,
        )
        self.assertEqual(1.0, util[0])
