import sys
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import UncertaintySampling, RandomSampling, DiscriminativeAL, CoreSet, TypiClust, Badge
from skactiveml.utils import call_func, MISSING_LABEL

import warnings
from tqdm import tqdm
import mlflow

def gen_seed(random_state:np.random.RandomState):
    return random_state.randint(0, 2**31)

def gen_random_state(random_state:np.random.RandomState):
    return np.random.RandomState(gen_seed(random_state))

def create_classifier(name, classes, random_state):
    return classifier_factory_functions[name](classes, random_state)

def create_query_strategy(name, random_state):
    return query_strategy_factory_functions[name](random_state)

if __name__ == '__main__':
    sys.path.append("/mnt/stud/home/jcheng/scikit-activeml/")

    mlp.rcParams["figure.facecolor"] = "white"
    warnings.filterwarnings("ignore")

    # CIFAR-10
    cifar10_X_train = np.load('./embedding_data/cifar10_dinov2_X_train.npy')
    cifar10_y_train_true = np.load('./embedding_data/cifar10_dinov2_y_train.npy')
    cifar10_X_test = np.load('./embedding_data/cifar10_dinov2_X_test.npy')
    cifar10_y_test_true = np.load('./embedding_data/cifar10_dinov2_y_test.npy')

    # CIFAR-100
    cifar100_X_train = np.load('./embedding_data/cifar100_dinov2_X_train.npy')
    cifar100_y_train_true = np.load('./embedding_data/cifar100_dinov2_y_train.npy')
    cifar100_X_test = np.load('./embedding_data/cifar100_dinov2_X_test.npy')
    cifar100_y_test_true = np.load('./embedding_data/cifar100_dinov2_y_test.npy')

    # Flowers-102
    flowers102_X_train = np.load('./embedding_data/flowers102_dinov2_X_train.npy')
    flowers102_y_train_true = np.load('./embedding_data/flowers102_dinov2_y_train.npy')
    flowers102_X_test = np.load('./embedding_data/flowers102_dinov2_X_test.npy')
    flowers102_y_test_true = np.load('./embedding_data/flowers102_dinov2_y_test.npy')
    dataset_classes = {
        "CIFAR10": 10,
        "CIFAR100": 100,
        "Flowers102": 102,
    }

    dataset = {
        "CIFAR10": {
            "X_train": cifar10_X_train,
            "y_train_true": cifar10_y_train_true,
            "X_test": cifar10_X_test,
            "y_test_true": cifar10_y_test_true,
        },
        "CIFAR100": {
            "X_train": cifar100_X_train,
            "y_train_true": cifar100_y_train_true,
            "X_test": cifar100_X_test,
            "y_test_true": cifar100_y_test_true,
        },
        "Flowers102": {
            "X_train": flowers102_X_train,
            "y_train_true": flowers102_y_train_true,
            "X_test": flowers102_X_test,
            "y_test_true": flowers102_y_test_true,
        },
    }

    master_random_state = np.random.RandomState(0)

    classifier_factory_functions = {
        'LogisticRegression': lambda classes, random_state: SklearnClassifier(
            LogisticRegression(),
            classes=classes,
            random_state=gen_seed(random_state)
        )
    }

    query_strategy_factory_functions = {
        'RandomSampling': lambda random_state: RandomSampling(random_state=gen_seed(random_state)),
        'UncertaintySampling': lambda random_state: UncertaintySampling(random_state=gen_seed(random_state)),
        'DiscriminativeAL': lambda random_state: DiscriminativeAL(random_state=gen_seed(random_state)),
        'CoreSet': lambda random_state: CoreSet(random_state=gen_seed(random_state)),
        'TypiClust': lambda random_state: TypiClust(random_state=gen_seed(random_state)),
        'Badge': lambda random_state: Badge(random_state=gen_seed(random_state))
    }

    n_reps = 1
    n_training_dataset = len(cifar10_X_train)
    # n_cycles = int(0.5 * n_training_dataset)
    n_cycles = 50
    classifier_names = classifier_factory_functions.keys()
    query_strategy_names = query_strategy_factory_functions.keys()
    dataset_names = dataset.keys()

    results = {}

    for data_name in dataset_names:
        data = dataset[data_name]
        data_classes = dataset_classes[data_name]
        X_train = data["X_train"]
        y_train_true = data["y_train_true"]
        X_test = data["X_test"]
        y_test_true = data["y_test_true"]

        for clf_name in classifier_names:
            for qs_name in query_strategy_names:
                accuracies = np.full((n_reps, n_cycles), np.nan)
                for i_rep in range(n_reps):
                    y_train = np.full(shape=y_train_true.shape, fill_value=MISSING_LABEL)

                    clf = create_classifier(clf_name, classes=np.arange(data_classes),
                                            random_state=gen_random_state(master_random_state))
                    qs = create_query_strategy(qs_name, random_state=gen_random_state(master_random_state))
                    clf.fit(X_train, y_train)

                    for c in tqdm(range(n_cycles),
                                  desc=f'Repeat {i_rep + 1} in {clf_name} with {qs_name} for {data_name}'):
                        query_idx = call_func(qs.query, X=X_train, y=y_train, batch_size=1, clf=clf, discriminator=clf)
                        y_train[query_idx] = y_train_true[query_idx]
                        clf.fit(X_train, y_train)
                        score = clf.score(X_test, y_test_true)
                        accuracies[i_rep, c] = score

                results[(data_name, clf_name, qs_name)] = accuracies

                mlflow.set_tracking_uri(uri="file:///mnt/stud/home/jcheng/scikit-activeml/tutorials/tracking")
                mlflow.set_experiment("Pool Evaluation with DINOv2")

                with mlflow.start_run():
                    for data_name in dataset_names:
                        for clf_name in classifier_names:
                            for qs_name in query_strategy_names:
                                key = (data_name, clf_name, qs_name)
                                result = results[key]
                                reshaped_result = result.reshape((-1, n_cycles))
                                errorbar_mean = np.mean(reshaped_result, axis=0)
                                mlflow.log_metric = (f'errorbar_mean for {qs_name} with {clf_name}', errorbar_mean)
                                errorbar_std = np.std(reshaped_result, axis=0)
                                mlflow.log_metric = (f'errorbar_std for {qs_name} with {clf_name}', errorbar_std)
                                plt.errorbar(np.arange(n_cycles), errorbar_mean, errorbar_std,
                                             label=f"({np.mean(errorbar_mean):.4f}) {qs_name}", alpha=0.5)
                            plt.title(clf_name)
                            plt.legend(loc='lower right')
                            plt.xlabel('cycle')
                            plt.ylabel('accuracy')
                            plt.show()