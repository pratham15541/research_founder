"""
Evidence Pipeline stages 2, 3, 5, 7.
Provides rigorous scientific verification before any gap is declared valid.
"""
from src.pipeline.methodological_verifier import MethodologicalVerifier
from src.pipeline.domain_verifier import DomainVerifier
from src.pipeline.prior_art_checker import PriorArtChecker
from src.pipeline.compatibility_reasoner import CompatibilityReasoner

__all__ = [
    "MethodologicalVerifier",
    "DomainVerifier",
    "PriorArtChecker",
    "CompatibilityReasoner",
]

