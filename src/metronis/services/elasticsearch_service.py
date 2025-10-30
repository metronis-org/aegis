'''
Elasticsearch Service - Advanced search for traces and evaluations
'''

from typing import List, Dict, Optional
from datetime import datetime
from elasticsearch import Elasticsearch, helpers
import structlog
from sqlalchemy.orm import Session

from metronis.db.models import TraceModel, EvaluationResultModel

logger = structlog.get_logger(__name__)


class ElasticsearchService:
    '''Manage Elasticsearch indexing and search.'''

    def __init__(self, es_url: str = 'http://localhost:9200'):
        self.client = Elasticsearch([es_url])
        self.traces_index = 'metronis_traces'
        self.evaluations_index = 'metronis_evaluations'

        logger.info('Elasticsearch service initialized', es_url=es_url)

    def create_indices(self):
        '''Create Elasticsearch indices with mappings.'''

        # Traces index
        traces_mapping = {
            'mappings': {
                'properties': {
                    'trace_id': {'type': 'keyword'},
                    'organization_id': {'type': 'keyword'},
                    'model': {'type': 'keyword'},
                    'domain': {'type': 'keyword'},
                    'input_text': {'type': 'text', 'analyzer': 'standard'},
                    'output_text': {'type': 'text', 'analyzer': 'standard'},
                    'created_at': {'type': 'date'},
                    'metadata': {'type': 'object', 'enabled': False},
                }
            }
        }

        if not self.client.indices.exists(index=self.traces_index):
            self.client.indices.create(index=self.traces_index, body=traces_mapping)
            logger.info('Traces index created', index=self.traces_index)

        # Evaluations index
        evaluations_mapping = {
            'mappings': {
                'properties': {
                    'evaluation_id': {'type': 'keyword'},
                    'trace_id': {'type': 'keyword'},
                    'organization_id': {'type': 'keyword'},
                    'overall_passed': {'type': 'boolean'},
                    'overall_severity': {'type': 'keyword'},
                    'total_issues': {'type': 'integer'},
                    'execution_time_ms': {'type': 'integer'},
                    'created_at': {'type': 'date'},
                }
            }
        }

        if not self.client.indices.exists(index=self.evaluations_index):
            self.client.indices.create(index=self.evaluations_index, body=evaluations_mapping)
            logger.info('Evaluations index created', index=self.evaluations_index)

    def index_trace(self, trace: TraceModel):
        '''Index a single trace.'''
        doc = {
            'trace_id': str(trace.trace_id),
            'organization_id': str(trace.organization_id),
            'model': trace.model,
            'domain': trace.domain,
            'input_text': trace.input_text,
            'output_text': trace.output_text,
            'created_at': trace.created_at.isoformat() if trace.created_at else None,
            'metadata': trace.metadata or {},
        }

        self.client.index(index=self.traces_index, id=str(trace.trace_id), document=doc)
        logger.debug('Trace indexed', trace_id=str(trace.trace_id))

    def index_evaluation(self, evaluation: EvaluationResultModel):
        '''Index a single evaluation.'''
        doc = {
            'evaluation_id': str(evaluation.evaluation_id),
            'trace_id': str(evaluation.trace_id),
            'organization_id': str(evaluation.organization_id),
            'overall_passed': evaluation.overall_passed,
            'overall_severity': evaluation.overall_severity,
            'total_issues': evaluation.total_issues,
            'execution_time_ms': evaluation.execution_time_ms,
            'created_at': evaluation.created_at.isoformat() if evaluation.created_at else None,
        }

        self.client.index(
            index=self.evaluations_index,
            id=str(evaluation.evaluation_id),
            document=doc
        )
        logger.debug('Evaluation indexed', evaluation_id=str(evaluation.evaluation_id))

    def bulk_index_traces(self, traces: List[TraceModel]):
        '''Bulk index multiple traces.'''
        actions = [
            {
                '_index': self.traces_index,
                '_id': str(trace.trace_id),
                '_source': {
                    'trace_id': str(trace.trace_id),
                    'organization_id': str(trace.organization_id),
                    'model': trace.model,
                    'domain': trace.domain,
                    'input_text': trace.input_text,
                    'output_text': trace.output_text,
                    'created_at': trace.created_at.isoformat() if trace.created_at else None,
                }
            }
            for trace in traces
        ]

        helpers.bulk(self.client, actions)
        logger.info('Traces bulk indexed', count=len(traces))

    def search_traces(
        self,
        organization_id: str,
        query: Optional[str] = None,
        domain: Optional[str] = None,
        model: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict]:
        '''Search traces with full-text and filters.'''

        # Build query
        must_clauses = [
            {'term': {'organization_id': organization_id}}
        ]

        if query:
            must_clauses.append({
                'multi_match': {
                    'query': query,
                    'fields': ['input_text', 'output_text'],
                    'type': 'best_fields',
                }
            })

        if domain:
            must_clauses.append({'term': {'domain': domain}})

        if model:
            must_clauses.append({'term': {'model': model}})

        if start_date or end_date:
            date_range = {}
            if start_date:
                date_range['gte'] = start_date.isoformat()
            if end_date:
                date_range['lte'] = end_date.isoformat()
            must_clauses.append({'range': {'created_at': date_range}})

        search_body = {
            'query': {
                'bool': {
                    'must': must_clauses
                }
            },
            'size': limit,
            'sort': [{'created_at': {'order': 'desc'}}]
        }

        result = self.client.search(index=self.traces_index, body=search_body)

        return [hit['_source'] for hit in result['hits']['hits']]

    def search_evaluations(
        self,
        organization_id: str,
        severity: Optional[str] = None,
        passed: Optional[bool] = None,
        limit: int = 100,
    ) -> List[Dict]:
        '''Search evaluations with filters.'''

        must_clauses = [
            {'term': {'organization_id': organization_id}}
        ]

        if severity:
            must_clauses.append({'term': {'overall_severity': severity}})

        if passed is not None:
            must_clauses.append({'term': {'overall_passed': passed}})

        search_body = {
            'query': {
                'bool': {
                    'must': must_clauses
                }
            },
            'size': limit,
            'sort': [{'created_at': {'order': 'desc'}}]
        }

        result = self.client.search(index=self.evaluations_index, body=search_body)

        return [hit['_source'] for hit in result['hits']['hits']]

    def get_aggregations(self, organization_id: str) -> Dict:
        '''Get aggregation statistics.'''

        agg_body = {
            'query': {
                'term': {'organization_id': organization_id}
            },
            'aggs': {
                'by_domain': {
                    'terms': {'field': 'domain', 'size': 10}
                },
                'by_model': {
                    'terms': {'field': 'model', 'size': 10}
                },
                'traces_over_time': {
                    'date_histogram': {
                        'field': 'created_at',
                        'calendar_interval': 'day'
                    }
                }
            },
            'size': 0
        }

        result = self.client.search(index=self.traces_index, body=agg_body)

        return {
            'by_domain': [
                {'key': bucket['key'], 'count': bucket['doc_count']}
                for bucket in result['aggregations']['by_domain']['buckets']
            ],
            'by_model': [
                {'key': bucket['key'], 'count': bucket['doc_count']}
                for bucket in result['aggregations']['by_model']['buckets']
            ],
            'over_time': [
                {'date': bucket['key_as_string'], 'count': bucket['doc_count']}
                for bucket in result['aggregations']['traces_over_time']['buckets']
            ]
        }
