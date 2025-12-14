"""Analytics module for game theory package."""

from .service import AnalyticsService
from .strategy import StrategyDetector, StrategyType, STRATEGY_DESCRIPTIONS
from .learning import LearningAnalyzer
from .equilibrium import EquilibriumAnalyzer
from .allocation import (
    AllocationAnalyzer,
    AllocationMetrics,
    SessionAllocationSummary,
    ComplianceMetrics,
    CrossSessionAnalyzer,
)
from .clustering import (
    StrategyClusterer,
    ClusterResult,
    StrategyArchetype,
    find_optimal_clusters,
)
from .sensitivity import (
    SensitivityResult,
    InteractionResult,
    OptimalParams,
    HyperparameterSensitivityAnalyzer,
)
from .meta_learning import (
    AdaptationMetrics,
    MemoryEffectResult,
    MultiLagMemoryAnalysis,
    LearningCurve,
    MetaStrategyAnalyzer,
)
from .personality import (
    ModelPersonalityProfile,
    FieldPreference,
    GameFingerprint,
    ModelPersonalityProfiler,
    # New granular analyzers
    BiasAnalyzer,
    BehaviorAnalyzer,
    GameFingerprintBuilder,
)
from .cross_game import (
    StrategyTransferMetrics,
    IntelligenceScores,
    CrossGameComparison,
    CrossGameComparativeAnalyzer,
)
from .role_service import (
    RoleAnalyticsService,
    RoleFilterParams,
    RoleStatistics,
    RoleGameBreakdown,
    RoleTimeline,
    DataSufficiency,
)

__all__ = [
    "AnalyticsService",
    "StrategyDetector",
    "StrategyType",
    "STRATEGY_DESCRIPTIONS",
    "LearningAnalyzer",
    "EquilibriumAnalyzer",
    # Allocation analytics
    "AllocationAnalyzer",
    "AllocationMetrics",
    "SessionAllocationSummary",
    "ComplianceMetrics",
    "CrossSessionAnalyzer",
    # Clustering
    "StrategyClusterer",
    "ClusterResult",
    "StrategyArchetype",
    "find_optimal_clusters",
    # Sensitivity analysis
    "SensitivityResult",
    "InteractionResult",
    "OptimalParams",
    "HyperparameterSensitivityAnalyzer",
    # Meta-learning
    "AdaptationMetrics",
    "MemoryEffectResult",
    "MultiLagMemoryAnalysis",
    "LearningCurve",
    "MetaStrategyAnalyzer",
    # Personality profiling
    "ModelPersonalityProfile",
    "FieldPreference",
    "GameFingerprint",
    "ModelPersonalityProfiler",
    "BiasAnalyzer",
    "BehaviorAnalyzer",
    "GameFingerprintBuilder",
    # Cross-game analysis
    "StrategyTransferMetrics",
    "IntelligenceScores",
    "CrossGameComparison",
    "CrossGameComparativeAnalyzer",
    # Role analytics
    "RoleAnalyticsService",
    "RoleFilterParams",
    "RoleStatistics",
    "RoleGameBreakdown",
    "RoleTimeline",
    "DataSufficiency",
]
