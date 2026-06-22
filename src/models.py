"""Klasyfikatory i metody resamplingu."""

from sklearn.ensemble import RandomForestClassifier  # noqa: F401
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE  # noqa: F401
from imblearn.under_sampling import RandomUnderSampler


def get_classifiers(random_state: int = 42) -> dict:
    return {
        "k-NN": KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
        "RF": RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=random_state, n_jobs=-1
        ),
        "GNB": GaussianNB(),
    }


def get_resamplers(random_state: int = 42) -> dict:
    return {
        "None": None,
        "ROS": RandomOverSampler(random_state=random_state),
        "RUS": RandomUnderSampler(random_state=random_state),
        "SMOTE": SMOTE(random_state=random_state),
    }
