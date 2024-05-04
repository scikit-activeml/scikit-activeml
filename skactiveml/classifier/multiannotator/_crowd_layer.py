from skactiveml.base import AnnotatorModelMixin
from skactiveml.classifier import SkorchClassifier


class CrowdLayer(SkorchClassifier, AnnotatorModelMixin):
    def __init__(self):

