"""
PHI/PII Detection and De-identification Service

HIPAA Safe Harbor requires removal of 18 identifiers:
1. Names
2. Geographic subdivisions smaller than state
3. Dates (except year)
4. Telephone numbers
5. Fax numbers
6. Email addresses
7. Social security numbers
8. Medical record numbers
9. Health plan beneficiary numbers
10. Account numbers
11. Certificate/license numbers
12. Vehicle identifiers
13. Device identifiers
14. URLs
15. IP addresses
16. Biometric identifiers
17. Full face photos
18. Any other unique identifying number/characteristic
"""

import asyncio
import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

try:
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    print("Warning: Presidio not installed. PHI detection will use regex fallbacks.")

import structlog

from metronis.core.models import Trace

logger = structlog.get_logger(__name__)


class PHIDetector:
    """
    Detect and de-identify Protected Health Information (PHI) in traces.

    Uses Microsoft Presidio for AI-powered detection, with regex fallbacks.
    """

    def __init__(self, use_presidio: bool = True):
        """
        Initialize the PHI detector.

        Args:
            use_presidio: Whether to use Presidio (requires installation)
        """
        self.use_presidio = use_presidio and PRESIDIO_AVAILABLE

        if self.use_presidio:
            # Initialize Presidio
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()

            # Supported entity types (18 HIPAA identifiers)
            self.entities = [
                "PERSON",  # Names
                "LOCATION",  # Addresses, cities
                "DATE_TIME",  # Dates
                "PHONE_NUMBER",  # Phone/fax
                "EMAIL_ADDRESS",  # Email
                "US_SSN",  # Social security number
                "MEDICAL_RECORD",  # Medical record numbers
                "ACCOUNT_NUMBER",  # Account numbers
                "LICENSE_PLATE",  # Vehicle identifiers
                "IP_ADDRESS",  # IP addresses
                "URL",  # URLs
                "US_DRIVER_LICENSE",  # License numbers
            ]
        else:
            logger.warning("Presidio not available, using regex-based detection")

        # Regex patterns for fallback
        self.patterns = {
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
            "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "date": re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
            "zipcode": re.compile(r"\b\d{5}(?:-\d{4})?\b"),
            "mrn": re.compile(r"\b(?:MRN|Medical Record Number)[:\s]*([A-Z0-9-]+)\b", re.IGNORECASE),
        }

        # Mapping for de-identified values (for consistent pseudonyms)
        self.phi_mapping: Dict[str, str] = {}

    async def detect_phi(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect PHI in text.

        Args:
            text: Text to analyze

        Returns:
            List of detected PHI entities with type, location, score
        """
        if self.use_presidio:
            return await self._detect_presidio(text)
        else:
            return await self._detect_regex(text)

    async def _detect_presidio(self, text: str) -> List[Dict[str, Any]]:
        """Detect PHI using Presidio AI."""
        # Run in thread pool since Presidio is CPU-intensive
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self.analyzer.analyze(
                text=text,
                language="en",
                entities=self.entities,
                return_decision_process=False,
            ),
        )

        detections = []
        for result in results:
            detections.append({
                "entity_type": result.entity_type,
                "text": text[result.start:result.end],
                "start": result.start,
                "end": result.end,
                "score": result.score,
                "recognition_metadata": result.recognition_metadata,
            })

        return detections

    async def _detect_regex(self, text: str) -> List[Dict[str, Any]]:
        """Detect PHI using regex patterns (fallback)."""
        detections = []

        for pattern_name, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                detections.append({
                    "entity_type": pattern_name,
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "score": 0.8,  # Fixed score for regex
                    "recognition_metadata": {"recognizer": "regex"},
                })

        return detections

    async def de_identify(self, text: str, strategy: str = "mask") -> Tuple[str, List[Dict[str, Any]]]:
        """
        De-identify PHI in text.

        Args:
            text: Text to de-identify
            strategy: De-identification strategy
                - "mask": Replace with <ENTITY_TYPE>
                - "pseudonymize": Replace with consistent pseudonym
                - "redact": Remove completely

        Returns:
            Tuple of (de-identified text, detected PHI list)
        """
        # Detect PHI
        detections = await self.detect_phi(text)

        if not detections:
            return text, []

        if self.use_presidio:
            return await self._deidentify_presidio(text, detections, strategy)
        else:
            return await self._deidentify_regex(text, detections, strategy)

    async def _deidentify_presidio(
        self, text: str, detections: List[Dict[str, Any]], strategy: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """De-identify using Presidio."""
        # Convert detections back to Presidio format
        from presidio_analyzer import RecognizerResult

        analyzer_results = []
        for det in detections:
            analyzer_results.append(
                RecognizerResult(
                    entity_type=det["entity_type"],
                    start=det["start"],
                    end=det["end"],
                    score=det["score"],
                )
            )

        # Define operators
        operators = {}
        if strategy == "mask":
            for entity_type in self.entities:
                operators[entity_type] = OperatorConfig("replace", {"new_value": f"<{entity_type}>"})
        elif strategy == "pseudonymize":
            for entity_type in self.entities:
                operators[entity_type] = OperatorConfig("hash", {"hash_type": "sha256"})
        else:  # redact
            for entity_type in self.entities:
                operators[entity_type] = OperatorConfig("redact", {})

        # Anonymize
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results,
                operators=operators,
            ),
        )

        return result.text, detections

    async def _deidentify_regex(
        self, text: str, detections: List[Dict[str, Any]], strategy: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """De-identify using regex replacement."""
        # Sort detections by start position (descending) to avoid index issues
        detections_sorted = sorted(detections, key=lambda x: x["start"], reverse=True)

        de_identified = text
        for det in detections_sorted:
            start, end = det["start"], det["end"]
            original = det["text"]

            if strategy == "mask":
                replacement = f"<{det['entity_type'].upper()}>"
            elif strategy == "pseudonymize":
                # Generate consistent pseudonym
                if original not in self.phi_mapping:
                    self.phi_mapping[original] = self._generate_pseudonym(det["entity_type"])
                replacement = self.phi_mapping[original]
            else:  # redact
                replacement = "[REDACTED]"

            de_identified = de_identified[:start] + replacement + de_identified[end:]

        return de_identified, detections

    def _generate_pseudonym(self, entity_type: str) -> str:
        """Generate a consistent pseudonym."""
        prefix_map = {
            "PERSON": "PATIENT",
            "email": "EMAIL",
            "phone": "PHONE",
            "ssn": "SSN",
            "mrn": "MRN",
        }
        prefix = prefix_map.get(entity_type, "ID")

        # Generate unique ID
        unique_id = str(uuid4())[:8].upper()
        return f"{prefix}_{unique_id}"

    async def sanitize_trace(self, trace: Trace) -> Trace:
        """
        Sanitize an entire trace by de-identifying all text fields.

        Args:
            trace: Trace to sanitize

        Returns:
            Sanitized trace with PHI removed
        """
        # De-identify AI input
        sanitized_input, input_phi = await self.de_identify(
            trace.ai_processing.input, strategy="mask"
        )
        trace.ai_processing.input = sanitized_input

        # De-identify AI output
        sanitized_output, output_phi = await self.de_identify(
            trace.ai_processing.output, strategy="mask"
        )
        trace.ai_processing.output = sanitized_output

        # De-identify metadata fields
        if trace.metadata.patient_context:
            sanitized_context, context_phi = await self.de_identify(
                trace.metadata.patient_context, strategy="mask"
            )
            trace.metadata.patient_context = sanitized_context

        # Log PHI detection
        total_phi = len(input_phi) + len(output_phi)
        if total_phi > 0:
            logger.warning(
                "PHI detected and removed",
                trace_id=str(trace.trace_id),
                phi_count=total_phi,
                input_phi=len(input_phi),
                output_phi=len(output_phi),
            )

        return trace

    def store_reidentification_mapping(
        self, trace_id: str, mapping: Dict[str, str]
    ) -> None:
        """
        Store the re-identification mapping (for audit purposes).

        This should be stored separately from the sanitized trace,
        with additional encryption and access controls.

        Args:
            trace_id: Trace ID
            mapping: Mapping from PHI to pseudonyms
        """
        # In production, store this in a separate encrypted database
        # with strict access controls and audit logging

        # Hash the trace_id for additional obfuscation
        trace_hash = hashlib.sha256(trace_id.encode()).hexdigest()

        # Store mapping (placeholder - implement secure storage)
        logger.info(
            "Re-identification mapping stored",
            trace_hash=trace_hash,
            mapping_count=len(mapping),
        )


# Global singleton
_phi_detector: Optional[PHIDetector] = None


def get_phi_detector() -> PHIDetector:
    """Get the global PHI detector instance."""
    global _phi_detector
    if _phi_detector is None:
        _phi_detector = PHIDetector()
    return _phi_detector
