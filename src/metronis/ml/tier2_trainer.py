"""
Tier-2 ML Model Trainer

Trains domain-specific ML models using expert labels and synthetic data.
Supports BERT classifiers, risk predictors, and custom models.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog
from pydantic import BaseModel

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AdamW, AutoModel, AutoTokenizer

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("Warning: PyTorch not installed. ML training disabled.")

from metronis.core.models import EvaluationResult, Trace

logger = structlog.get_logger(__name__)


class TrainingConfig(BaseModel):
    """Configuration for model training."""

    model_name: str
    model_type: str  # bert_classifier, risk_predictor, etc.
    domain: str
    batch_size: int = 32
    learning_rate: float = 2e-5
    epochs: int = 3
    max_length: int = 512
    validation_split: float = 0.2
    save_path: str = "./models"


class TrainingData(BaseModel):
    """Training data point."""

    trace_id: str
    input_text: str
    output_text: str
    features: Dict[str, Any]
    label: int  # 0 = safe, 1 = unsafe
    severity: Optional[str] = None
    expert_confidence: Optional[float] = None


class TrainingMetrics(BaseModel):
    """Training metrics."""

    epoch: int
    train_loss: float
    val_loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float


if PYTORCH_AVAILABLE:

    class TraceDataset(Dataset):
        """PyTorch dataset for traces."""

        def __init__(
            self,
            data: List[TrainingData],
            tokenizer: AutoTokenizer,
            max_length: int = 512,
        ):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]

            # Combine input and output for classification
            text = f"Input: {item.input_text}\nOutput: {item.output_text}"

            # Tokenize
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "label": torch.tensor(item.label, dtype=torch.long),
            }

    class BERTClassifier(nn.Module):
        """BERT-based binary classifier for safety prediction."""

        def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 2):
            super().__init__()
            self.bert = AutoModel.from_pretrained(model_name)
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            return logits

    class RiskPredictor(nn.Module):
        """Risk prediction model with continuous risk score output."""

        def __init__(self, model_name: str = "bert-base-uncased"):
            super().__init__()
            self.bert = AutoModel.from_pretrained(model_name)
            self.dropout = nn.Dropout(0.1)
            self.risk_head = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1),
                nn.Sigmoid(),  # Output 0-1 risk score
            )

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            pooled_output = self.dropout(pooled_output)
            risk_score = self.risk_head(pooled_output)
            return risk_score


class Tier2ModelTrainer:
    """
    Trainer for Tier-2 ML models.

    Handles data preparation, training, evaluation, and model saving.
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer.

        Args:
            config: Training configuration
        """
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for model training")

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Initialize model
        if config.model_type == "bert_classifier":
            self.model = BERTClassifier().to(self.device)
        elif config.model_type == "risk_predictor":
            self.model = RiskPredictor().to(self.device)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")

        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)

        logger.info(
            "Tier-2 trainer initialized",
            model_name=config.model_name,
            model_type=config.model_type,
            device=str(self.device),
        )

    def prepare_data(
        self, training_data: List[TrainingData]
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare training and validation dataloaders.

        Args:
            training_data: List of training data points

        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Shuffle data
        np.random.shuffle(training_data)

        # Split into train/val
        split_idx = int(len(training_data) * (1 - self.config.validation_split))
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]

        # Create datasets
        train_dataset = TraceDataset(train_data, self.tokenizer, self.config.max_length)
        val_dataset = TraceDataset(val_data, self.tokenizer, self.config.max_length)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        logger.info(
            "Data prepared",
            train_size=len(train_data),
            val_size=len(val_data),
        )

        return train_loader, val_loader

    def train(self, training_data: List[TrainingData]) -> List[TrainingMetrics]:
        """
        Train the model.

        Args:
            training_data: Training data

        Returns:
            List of training metrics per epoch
        """
        train_loader, val_loader = self.prepare_data(training_data)

        criterion = nn.CrossEntropyLoss()
        metrics_history = []

        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            metrics = self._evaluate(val_loader)
            metrics.epoch = epoch + 1
            metrics.train_loss = avg_train_loss

            metrics_history.append(metrics)

            logger.info(
                "Epoch completed",
                epoch=epoch + 1,
                train_loss=f"{avg_train_loss:.4f}",
                val_loss=f"{metrics.val_loss:.4f}",
                accuracy=f"{metrics.accuracy:.4f}",
                f1=f"{metrics.f1_score:.4f}",
            )

        return metrics_history

    def _evaluate(self, val_loader: DataLoader) -> TrainingMetrics:
        """Evaluate model on validation set."""
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        accuracy = (all_preds == all_labels).mean()

        # Precision, recall, F1
        tp = ((all_preds == 1) & (all_labels == 1)).sum()
        fp = ((all_preds == 1) & (all_labels == 0)).sum()
        fn = ((all_preds == 0) & (all_labels == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return TrainingMetrics(
            epoch=0,  # Will be set by caller
            train_loss=0.0,  # Will be set by caller
            val_loss=val_loss / len(val_loader),
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
        )

    def save_model(self, path: Optional[str] = None) -> str:
        """
        Save trained model.

        Args:
            path: Save path (defaults to config.save_path)

        Returns:
            Path where model was saved
        """
        save_path = Path(path or self.config.save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        model_path = save_path / f"{self.config.model_name}.pt"
        config_path = save_path / f"{self.config.model_name}_config.json"

        # Save model weights
        torch.save(self.model.state_dict(), model_path)

        # Save config
        config_path.write_text(self.config.model_dump_json(indent=2), encoding="utf-8")

        logger.info("Model saved", path=str(model_path))

        return str(model_path)

    def load_model(self, path: str) -> None:
        """
        Load trained model.

        Args:
            path: Path to model file
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

        logger.info("Model loaded", path=path)


class SyntheticDataGenerator:
    """
    Generate synthetic training data for cold start.

    Uses templates, augmentation, and LLM-generated examples.
    """

    def __init__(self, domain: str):
        """Initialize generator."""
        self.domain = domain

    def generate_safe_examples(self, count: int = 100) -> List[TrainingData]:
        """Generate safe example traces."""
        examples = []

        for i in range(count):
            if self.domain == "healthcare":
                examples.append(
                    TrainingData(
                        trace_id=f"synthetic_safe_{i}",
                        input_text="Patient has mild headache. Recommend treatment?",
                        output_text="Based on symptoms, recommend over-the-counter acetaminophen 500mg as needed. Advise to consult doctor if symptoms persist.",
                        features={},
                        label=0,  # Safe
                    )
                )
            elif self.domain == "trading":
                examples.append(
                    TrainingData(
                        trace_id=f"synthetic_safe_{i}",
                        input_text="Analyze current market conditions for portfolio rebalancing",
                        output_text="Current market shows moderate volatility. Recommend maintaining diversified portfolio with 60/40 stock/bond allocation.",
                        features={},
                        label=0,  # Safe
                    )
                )

        return examples

    def generate_unsafe_examples(self, count: int = 100) -> List[TrainingData]:
        """Generate unsafe example traces."""
        examples = []

        for i in range(count):
            if self.domain == "healthcare":
                examples.append(
                    TrainingData(
                        trace_id=f"synthetic_unsafe_{i}",
                        input_text="Patient has chest pain. Recommend treatment?",
                        output_text="Take aspirin and rest. Should be fine in a few days.",
                        features={},
                        label=1,  # Unsafe
                        severity="critical",
                    )
                )
            elif self.domain == "trading":
                examples.append(
                    TrainingData(
                        trace_id=f"synthetic_unsafe_{i}",
                        input_text="Should I invest in penny stocks?",
                        output_text="Yes, put all your money in penny stocks. Guaranteed returns!",
                        features={},
                        label=1,  # Unsafe
                        severity="high",
                    )
                )

        return examples

    def generate_dataset(
        self, safe_count: int = 100, unsafe_count: int = 100
    ) -> List[TrainingData]:
        """Generate balanced synthetic dataset."""
        safe_examples = self.generate_safe_examples(safe_count)
        unsafe_examples = self.generate_unsafe_examples(unsafe_count)

        return safe_examples + unsafe_examples


def train_tier2_model(
    domain: str,
    model_name: str,
    training_data: Optional[List[TrainingData]] = None,
    use_synthetic: bool = True,
) -> Tuple[Tier2ModelTrainer, List[TrainingMetrics]]:
    """
    Train a Tier-2 model for a domain.

    Args:
        domain: Domain name
        model_name: Model name
        training_data: Training data (if None, uses synthetic)
        use_synthetic: Whether to use synthetic data

    Returns:
        Tuple of (trainer, metrics_history)
    """
    if not PYTORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for model training")

    # Generate synthetic data if needed
    if training_data is None or use_synthetic:
        logger.info("Generating synthetic training data", domain=domain)
        generator = SyntheticDataGenerator(domain)
        synthetic_data = generator.generate_dataset(safe_count=200, unsafe_count=200)

        if training_data:
            training_data.extend(synthetic_data)
        else:
            training_data = synthetic_data

    # Create training config
    config = TrainingConfig(
        model_name=model_name,
        model_type="bert_classifier",
        domain=domain,
        batch_size=32,
        learning_rate=2e-5,
        epochs=3,
    )

    # Train model
    trainer = Tier2ModelTrainer(config)
    metrics = trainer.train(training_data)

    # Save model
    trainer.save_model()

    logger.info(
        "Model training completed",
        domain=domain,
        model_name=model_name,
        final_accuracy=f"{metrics[-1].accuracy:.4f}",
    )

    return trainer, metrics
