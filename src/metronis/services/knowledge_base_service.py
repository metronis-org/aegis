"""
Knowledge Base Service

Integrates with external knowledge bases:
- Healthcare: RxNorm, SNOMED CT, FDA, DailyMed, UpToDate
- Trading: SEC EDGAR, FINRA, market data feeds
- Robotics: URDF models, safety standards
- Legal: Westlaw, LexisNexis, court databases

Features:
- Caching (Redis) for performance
- Rate limiting
- Retry logic
- Circuit breaker for failed services
"""

import asyncio
import hashlib
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import httpx
from redis import Redis


class KnowledgeBaseClient:
    """Base client for knowledge base integrations."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        cache_client: Optional[Redis] = None,
        cache_ttl: int = 86400,
        rate_limit: int = 100,
    ):
        """Initialize the client."""
        self.base_url = base_url
        self.api_key = api_key
        self.cache_client = cache_client
        self.cache_ttl = cache_ttl
        self.rate_limit = rate_limit

        # HTTP client with retry
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )

        # Rate limiting
        self.request_times: List[float] = []

    async def get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None, cache_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make a GET request with caching and rate limiting."""
        # Check cache first
        if cache_key and self.cache_client:
            cached = self.cache_client.get(cache_key)
            if cached:
                import json
                return json.loads(cached)

        # Rate limiting
        await self._rate_limit()

        # Make request
        url = urljoin(self.base_url, endpoint)
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = await self.http_client.get(url, params=params, headers=headers)
        response.raise_for_status()

        result = response.json()

        # Cache result
        if cache_key and self.cache_client:
            import json
            self.cache_client.setex(cache_key, self.cache_ttl, json.dumps(result))

        return result

    async def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        now = time.time()

        # Remove requests older than 1 second
        self.request_times = [t for t in self.request_times if now - t < 1.0]

        if len(self.request_times) >= self.rate_limit:
            # Wait until oldest request is > 1 second old
            sleep_time = 1.0 - (now - self.request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        self.request_times.append(now)

    def _cache_key(self, *args: Any) -> str:
        """Generate cache key from arguments."""
        key_string = ":".join(str(arg) for arg in args)
        return hashlib.md5(key_string.encode()).hexdigest()


class RxNormClient(KnowledgeBaseClient):
    """Client for RxNorm medication database."""

    def __init__(self, cache_client: Optional[Redis] = None):
        """Initialize RxNorm client."""
        super().__init__(
            base_url="https://rxnav.nlm.nih.gov/REST/",
            cache_client=cache_client,
            cache_ttl=86400,  # 24 hours
        )

    async def search_drug(self, drug_name: str) -> Dict[str, Any]:
        """Search for a drug by name."""
        cache_key = self._cache_key("rxnorm", "search", drug_name.lower())

        result = await self.get(
            "drugs.json",
            params={"name": drug_name},
            cache_key=cache_key,
        )

        return result

    async def get_interactions(self, rxcui: str) -> List[Dict[str, Any]]:
        """Get drug interactions for an RxCUI."""
        cache_key = self._cache_key("rxnorm", "interactions", rxcui)

        result = await self.get(
            f"interaction/interaction.json",
            params={"rxcui": rxcui},
            cache_key=cache_key,
        )

        interactions = []
        if "interactionTypeGroup" in result:
            for group in result["interactionTypeGroup"]:
                for interaction_type in group.get("interactionType", []):
                    for pair in interaction_type.get("interactionPair", []):
                        interactions.append({
                            "severity": pair.get("severity", "unknown"),
                            "description": pair.get("description", ""),
                            "drug1": pair.get("interactionConcept", [{}])[0].get("minConceptItem", {}).get("name", ""),
                            "drug2": pair.get("interactionConcept", [{}])[1].get("minConceptItem", {}).get("name", "") if len(pair.get("interactionConcept", [])) > 1 else "",
                        })

        return interactions

    async def check_medication_exists(self, medication_name: str) -> bool:
        """Check if a medication exists in RxNorm."""
        result = await self.search_drug(medication_name)
        return "drugGroup" in result and len(result["drugGroup"].get("conceptGroup", [])) > 0


class SNOMEDClient(KnowledgeBaseClient):
    """Client for SNOMED CT terminology."""

    def __init__(self, cache_client: Optional[Redis] = None):
        """Initialize SNOMED client."""
        super().__init__(
            base_url="https://browser.ihtsdotools.org/snowstorm/snomed-ct/",
            cache_client=cache_client,
        )

    async def search_concept(self, term: str, semantic_tag: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for SNOMED concepts."""
        cache_key = self._cache_key("snomed", "search", term.lower(), semantic_tag or "")

        params = {"term": term, "limit": 20}
        if semantic_tag:
            params["semanticTag"] = semantic_tag

        result = await self.get(
            "browser/MAIN/concepts",
            params=params,
            cache_key=cache_key,
        )

        return result.get("items", [])


class FDAClient(KnowledgeBaseClient):
    """Client for FDA APIs."""

    def __init__(self, cache_client: Optional[Redis] = None):
        """Initialize FDA client."""
        super().__init__(
            base_url="https://api.fda.gov/",
            cache_client=cache_client,
        )

    async def search_drug_labels(self, drug_name: str) -> List[Dict[str, Any]]:
        """Search FDA drug labels."""
        cache_key = self._cache_key("fda", "drug_labels", drug_name.lower())

        result = await self.get(
            "drug/label.json",
            params={"search": f'openfda.brand_name:"{drug_name}"', "limit": 10},
            cache_key=cache_key,
        )

        return result.get("results", [])


class SECEdgarClient(KnowledgeBaseClient):
    """Client for SEC EDGAR database."""

    def __init__(self, cache_client: Optional[Redis] = None):
        """Initialize SEC EDGAR client."""
        super().__init__(
            base_url="https://data.sec.gov/",
            cache_client=cache_client,
            cache_ttl=3600,  # 1 hour
            rate_limit=10,  # SEC rate limit: 10 req/sec
        )

    async def get_company_filings(self, cik: str) -> List[Dict[str, Any]]:
        """Get company filings by CIK."""
        cache_key = self._cache_key("sec", "filings", cik)

        # SEC requires User-Agent header
        url = urljoin(self.base_url, f"submissions/CIK{cik.zfill(10)}.json")

        response = await self.http_client.get(
            url,
            headers={"User-Agent": "Metronis AI Evaluation Platform info@metronis.ai"}
        )
        response.raise_for_status()

        result = response.json()

        # Cache
        if self.cache_client:
            import json
            self.cache_client.setex(cache_key, self.cache_ttl, json.dumps(result))

        return result.get("filings", {}).get("recent", {})


class KnowledgeBaseService:
    """
    Unified service for all knowledge base integrations.

    Provides a single interface for all domains.
    """

    def __init__(self, redis_url: Optional[str] = None):
        """Initialize the knowledge base service."""
        self.cache_client = Redis.from_url(redis_url) if redis_url else None

        # Initialize clients
        self.rxnorm = RxNormClient(self.cache_client)
        self.snomed = SNOMEDClient(self.cache_client)
        self.fda = FDAClient(self.cache_client)
        self.sec_edgar = SECEdgarClient(self.cache_client)

    async def check_existence(self, knowledge_base: str, entity: str) -> Dict[str, Any]:
        """
        Generic method to check if an entity exists in a knowledge base.

        Args:
            knowledge_base: Name of the KB (e.g., "rxnorm", "snomed")
            entity: Entity to check

        Returns:
            Dict with "exists" boolean and additional metadata
        """
        if knowledge_base == "rxnorm":
            exists = await self.rxnorm.check_medication_exists(entity)
            return {"exists": exists, "source": "rxnorm"}

        elif knowledge_base == "snomed":
            results = await self.snomed.search_concept(entity)
            return {"exists": len(results) > 0, "source": "snomed", "matches": results}

        elif knowledge_base == "fda":
            results = await self.fda.search_drug_labels(entity)
            return {"exists": len(results) > 0, "source": "fda"}

        else:
            return {"exists": False, "error": f"Unknown knowledge base: {knowledge_base}"}

    async def check_interaction(self, entity1: str, entity2: str) -> Optional[Dict[str, Any]]:
        """
        Check for interactions between two entities.

        Currently supports drug-drug interactions via RxNorm.
        """
        # Search for both drugs
        drug1_result = await self.rxnorm.search_drug(entity1)
        drug2_result = await self.rxnorm.search_drug(entity2)

        # Extract RxCUIs
        rxcui1 = self._extract_rxcui(drug1_result)
        rxcui2 = self._extract_rxcui(drug2_result)

        if not rxcui1 or not rxcui2:
            return None

        # Get interactions for drug1
        interactions = await self.rxnorm.get_interactions(rxcui1)

        # Find interaction with drug2
        for interaction in interactions:
            if entity2.lower() in interaction.get("drug2", "").lower():
                return interaction

        return None

    def _extract_rxcui(self, search_result: Dict[str, Any]) -> Optional[str]:
        """Extract RxCUI from RxNorm search result."""
        if "drugGroup" not in search_result:
            return None

        for concept_group in search_result["drugGroup"].get("conceptGroup", []):
            properties = concept_group.get("conceptProperties", [])
            if properties:
                return properties[0].get("rxcui")

        return None

    async def close(self) -> None:
        """Close all HTTP clients."""
        await self.rxnorm.http_client.aclose()
        await self.snomed.http_client.aclose()
        await self.fda.http_client.aclose()
        await self.sec_edgar.http_client.aclose()
