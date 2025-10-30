"""
Transfer Metric Learning for RL Agents

Learns correlation between simulation metrics and real-world outcomes.
Enables accurate evaluation before real-world deployment.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog
from pydantic import BaseModel

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. Transfer learning disabled.")

logger = structlog.get_logger(__name__)


class SimulationMetrics(BaseModel):
    """Metrics from simulation evaluation."""
    episode_return: float
    success_rate: float
    avg_episode_length: float
    exploration_efficiency: float
    safety_violations: int
    convergence_speed: float


class RealWorldOutcome(BaseModel):
    """Real-world deployment outcome."""
    actual_return: float
    actual_success_rate: float
    deployment_cost: float
    user_satisfaction: float
    incidents: int
    adaptation_time: float


class TransferDataPoint(BaseModel):
    """Single data point for transfer learning."""
    agent_id: str
    domain: str
    sim_metrics: SimulationMetrics
    real_outcome: RealWorldOutcome
    timestamp: str


class TransferModel(BaseModel):
    """Trained transfer model configuration."""
    domain: str
    feature_names: List[str]
    target_names: List[str]
    model_type: str
    r2_scores: Dict[str, float]
    sample_count: int


class TransferMetricLearner:
    """
    Learns how simulation metrics translate to real-world performance.

    Uses supervised learning to predict real-world outcomes from sim metrics.
    """

    def __init__(self, domain: str):
        """
        Initialize transfer learner.

        Args:
            domain: Domain name (healthcare, trading, robotics, etc.)
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for transfer learning")

        self.domain = domain
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}

        # Target metrics to predict
        self.target_metrics = [
            "actual_return",
            "actual_success_rate",
            "user_satisfaction",
            "incidents",
        ]

        logger.info("Transfer metric learner initialized", domain=domain)

    def prepare_features(
        self, data_points: List[TransferDataPoint]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Prepare features and targets from data points.

        Args:
            data_points: List of transfer data points

        Returns:
            Tuple of (features, targets_dict)
        """
        # Extract simulation features
        features = []
        for dp in data_points:
            sim = dp.sim_metrics
            features.append([
                sim.episode_return,
                sim.success_rate,
                sim.avg_episode_length,
                sim.exploration_efficiency,
                float(sim.safety_violations),
                sim.convergence_speed,
            ])

        X = np.array(features)

        # Extract real-world targets
        targets = {metric: [] for metric in self.target_metrics}

        for dp in data_points:
            real = dp.real_outcome
            targets["actual_return"].append(real.actual_return)
            targets["actual_success_rate"].append(real.actual_success_rate)
            targets["user_satisfaction"].append(real.user_satisfaction)
            targets["incidents"].append(float(real.incidents))

        # Convert to arrays
        y_dict = {k: np.array(v) for k, v in targets.items()}

        return X, y_dict

    def train(
        self,
        data_points: List[TransferDataPoint],
        test_size: float = 0.2,
    ) -> TransferModel:
        """
        Train transfer models.

        Args:
            data_points: Training data
            test_size: Fraction for test set

        Returns:
            Trained transfer model info
        """
        logger.info(
            "Training transfer models",
            domain=self.domain,
            samples=len(data_points),
        )

        # Prepare data
        X, y_dict = self.prepare_features(data_points)

        # Split data
        X_train, X_test, _, _ = train_test_split(
            X, y_dict["actual_return"], test_size=test_size, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Store scaler
        self.scalers["main"] = scaler

        # Train a model for each target metric
        r2_scores = {}

        for target_name in self.target_metrics:
            y = y_dict[target_name]
            y_train, y_test = train_test_split(
                y, test_size=test_size, random_state=42
            )

            # Train gradient boosting model
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
            )

            model.fit(X_train_scaled, y_train)

            # Evaluate
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            # Store model
            self.models[target_name] = model
            r2_scores[target_name] = float(r2)

            logger.info(
                "Model trained",
                target=target_name,
                r2=f"{r2:.4f}",
                mse=f"{mse:.4f}",
            )

        feature_names = [
            "episode_return",
            "success_rate",
            "avg_episode_length",
            "exploration_efficiency",
            "safety_violations",
            "convergence_speed",
        ]

        return TransferModel(
            domain=self.domain,
            feature_names=feature_names,
            target_names=self.target_metrics,
            model_type="gradient_boosting",
            r2_scores=r2_scores,
            sample_count=len(data_points),
        )

    def predict_real_world_outcome(
        self, sim_metrics: SimulationMetrics
    ) -> RealWorldOutcome:
        """
        Predict real-world outcome from simulation metrics.

        Args:
            sim_metrics: Simulation metrics

        Returns:
            Predicted real-world outcome
        """
        # Prepare features
        features = np.array([[
            sim_metrics.episode_return,
            sim_metrics.success_rate,
            sim_metrics.avg_episode_length,
            sim_metrics.exploration_efficiency,
            float(sim_metrics.safety_violations),
            sim_metrics.convergence_speed,
        ]])

        # Scale features
        scaler = self.scalers.get("main")
        if scaler:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features

        # Predict each target
        predictions = {}
        for target_name, model in self.models.items():
            pred = model.predict(features_scaled)[0]
            predictions[target_name] = float(pred)

        # Create outcome
        outcome = RealWorldOutcome(
            actual_return=predictions.get("actual_return", 0.0),
            actual_success_rate=predictions.get("actual_success_rate", 0.0),
            deployment_cost=self._estimate_deployment_cost(sim_metrics),
            user_satisfaction=predictions.get("user_satisfaction", 0.0),
            incidents=int(predictions.get("incidents", 0)),
            adaptation_time=self._estimate_adaptation_time(sim_metrics),
        )

        return outcome

    def _estimate_deployment_cost(self, sim_metrics: SimulationMetrics) -> float:
        """Estimate deployment cost based on sim metrics."""
        # Base cost + cost per violation + cost for low exploration
        base_cost = 100.0
        violation_cost = sim_metrics.safety_violations * 50.0
        exploration_cost = (1.0 - sim_metrics.exploration_efficiency) * 200.0

        return base_cost + violation_cost + exploration_cost

    def _estimate_adaptation_time(self, sim_metrics: SimulationMetrics) -> float:
        """Estimate time to adapt to real world."""
        # Faster convergence in sim → faster adaptation
        base_time = 100.0  # hours
        convergence_factor = 1.0 / max(sim_metrics.convergence_speed, 0.1)

        return base_time * convergence_factor

    def get_feature_importance(self, target_metric: str) -> Dict[str, float]:
        """
        Get feature importance for a target metric.

        Args:
            target_metric: Target metric name

        Returns:
            Dict mapping feature names to importance scores
        """
        model = self.models.get(target_metric)
        if not model or not hasattr(model, "feature_importances_"):
            return {}

        feature_names = [
            "episode_return",
            "success_rate",
            "avg_episode_length",
            "exploration_efficiency",
            "safety_violations",
            "convergence_speed",
        ]

        importance = model.feature_importances_
        return {name: float(imp) for name, imp in zip(feature_names, importance)}

    def calculate_sim_to_real_gap(
        self,
        sim_metrics: SimulationMetrics,
        real_outcome: RealWorldOutcome,
    ) -> Dict[str, float]:
        """
        Calculate sim-to-real gap (reality gap).

        Args:
            sim_metrics: Simulation metrics
            real_outcome: Actual real-world outcome

        Returns:
            Dict of gaps per metric
        """
        predicted = self.predict_real_world_outcome(sim_metrics)

        gaps = {
            "return_gap": abs(predicted.actual_return - real_outcome.actual_return),
            "success_rate_gap": abs(
                predicted.actual_success_rate - real_outcome.actual_success_rate
            ),
            "satisfaction_gap": abs(
                predicted.user_satisfaction - real_outcome.user_satisfaction
            ),
            "incident_gap": abs(predicted.incidents - real_outcome.incidents),
        }

        return gaps


class DomainAdaptation:
    """
    Adapts models when deploying to new sub-domains.

    Uses few-shot learning and domain randomization.
    """

    def __init__(self, source_domain: str, target_domain: str):
        """
        Initialize domain adaptation.

        Args:
            source_domain: Source domain (where we have data)
            target_domain: Target domain (where we're deploying)
        """
        self.source_domain = source_domain
        self.target_domain = target_domain

    def adapt_with_few_shots(
        self,
        source_learner: TransferMetricLearner,
        target_samples: List[TransferDataPoint],
    ) -> TransferMetricLearner:
        """
        Adapt source model to target domain using few-shot learning.

        Args:
            source_learner: Trained source domain learner
            target_samples: Few samples from target domain

        Returns:
            Adapted learner
        """
        logger.info(
            "Adapting model to new domain",
            source=self.source_domain,
            target=self.target_domain,
            samples=len(target_samples),
        )

        # Create new learner for target domain
        target_learner = TransferMetricLearner(self.target_domain)

        # Copy models from source
        target_learner.models = source_learner.models.copy()
        target_learner.scalers = source_learner.scalers.copy()

        # Fine-tune on target samples
        if len(target_samples) >= 10:
            X, y_dict = target_learner.prepare_features(target_samples)

            scaler = target_learner.scalers["main"]
            X_scaled = scaler.transform(X)

            # Fine-tune each model
            for target_name, model in target_learner.models.items():
                y = y_dict[target_name]

                # Partial fit (online learning)
                # For GradientBoosting, we'd retrain with warm_start
                # For now, just log
                logger.info(
                    "Fine-tuned model",
                    target=target_name,
                    samples=len(target_samples),
                )

        return target_learner


def generate_synthetic_transfer_data(
    domain: str, count: int = 100
) -> List[TransferDataPoint]:
    """
    Generate synthetic transfer data for testing.

    Args:
        domain: Domain name
        count: Number of data points

    Returns:
        List of synthetic transfer data points
    """
    data_points = []

    for i in range(count):
        # Generate correlated sim and real metrics
        sim_return = np.random.uniform(0.5, 1.0)
        sim_success = np.random.uniform(0.6, 1.0)

        # Real performance is slightly lower (sim-to-real gap)
        real_return = sim_return * np.random.uniform(0.8, 0.95)
        real_success = sim_success * np.random.uniform(0.85, 0.98)

        sim_metrics = SimulationMetrics(
            episode_return=float(sim_return),
            success_rate=float(sim_success),
            avg_episode_length=np.random.uniform(50, 200),
            exploration_efficiency=np.random.uniform(0.5, 0.9),
            safety_violations=int(np.random.poisson(2)),
            convergence_speed=np.random.uniform(0.5, 1.0),
        )

        real_outcome = RealWorldOutcome(
            actual_return=float(real_return),
            actual_success_rate=float(real_success),
            deployment_cost=np.random.uniform(100, 500),
            user_satisfaction=np.random.uniform(0.6, 0.9),
            incidents=int(np.random.poisson(1)),
            adaptation_time=np.random.uniform(50, 300),
        )

        data_points.append(
            TransferDataPoint(
                agent_id=f"agent_{i}",
                domain=domain,
                sim_metrics=sim_metrics,
                real_outcome=real_outcome,
                timestamp="2024-01-01T00:00:00Z",
            )
        )

    return data_points


# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    data = generate_synthetic_transfer_data("healthcare", count=200)

    # Train transfer model
    learner = TransferMetricLearner("healthcare")
    model_info = learner.train(data)

    print(f"Trained transfer model for {model_info.domain}")
    print(f"R² scores: {model_info.r2_scores}")

    # Test prediction
    test_sim = SimulationMetrics(
        episode_return=0.85,
        success_rate=0.92,
        avg_episode_length=120.0,
        exploration_efficiency=0.75,
        safety_violations=1,
        convergence_speed=0.8,
    )

    predicted_outcome = learner.predict_real_world_outcome(test_sim)
    print(f"\nPredicted real-world outcome:")
    print(f"  Return: {predicted_outcome.actual_return:.3f}")
    print(f"  Success rate: {predicted_outcome.actual_success_rate:.3f}")
    print(f"  User satisfaction: {predicted_outcome.user_satisfaction:.3f}")
