/**
 * API Client for Metronis Aegis
 */

import axios, { AxiosInstance } from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export interface Trace {
  trace_id: string;
  model: string;
  input: string;
  output: string;
  domain: string;
  created_at: string;
}

export interface Evaluation {
  evaluation_id: string;
  trace_id: string;
  overall_passed: boolean;
  overall_severity: string;
  total_issues: number;
  execution_time_ms: number;
  created_at: string;
}

export interface UsageSummary {
  organization_id: string;
  start_date: string;
  end_date: string;
  metrics: Record<string, { count: number; cost: number }>;
  total_cost: number;
}

class MetronisAPI {
  private client: AxiosInstance;
  private apiKey: string | null = null;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add auth interceptor
    this.client.interceptors.request.use((config) => {
      if (this.apiKey) {
        config.headers.Authorization = `Bearer ${this.apiKey}`;
      }
      return config;
    });
  }

  setApiKey(apiKey: string) {
    this.apiKey = apiKey;
  }

  // Traces
  async createTrace(data: {
    model: string;
    input: string;
    output: string;
    domain: string;
  }): Promise<Trace> {
    const response = await this.client.post('/api/v1/traces', data);
    return response.data;
  }

  async listTraces(params?: {
    domain?: string;
    limit?: number;
    offset?: number;
  }): Promise<Trace[]> {
    const response = await this.client.get('/api/v1/traces', { params });
    return response.data;
  }

  async getTrace(traceId: string): Promise<Trace> {
    const response = await this.client.get(`/api/v1/traces/${traceId}`);
    return response.data;
  }

  // Evaluations
  async listEvaluations(params?: {
    trace_id?: string;
    limit?: number;
  }): Promise<Evaluation[]> {
    const response = await this.client.get('/api/v1/evaluations', { params });
    return response.data;
  }

  async getEvaluation(evaluationId: string): Promise<Evaluation> {
    const response = await this.client.get(`/api/v1/evaluations/${evaluationId}`);
    return response.data;
  }

  // Billing
  async getUsageSummary(startDate?: string, endDate?: string): Promise<UsageSummary> {
    const response = await this.client.get('/api/v1/billing/usage/summary', {
      params: { start_date: startDate, end_date: endDate },
    });
    return response.data;
  }

  // Compliance
  async getFDAReport(startDate?: string, endDate?: string): Promise<any> {
    const response = await this.client.get('/api/v1/compliance/fda-tplc', {
      params: { start_date: startDate, end_date: endDate },
    });
    return response.data;
  }

  async getHIPAAReport(startDate?: string, endDate?: string): Promise<any> {
    const response = await this.client.get('/api/v1/compliance/hipaa', {
      params: { start_date: startDate, end_date: endDate },
    });
    return response.data;
  }

  // WebSocket
  connectWebSocket(apiKey: string): WebSocket {
    const wsUrl = API_BASE_URL.replace('http', 'ws') + `/ws/traces?api_key=${apiKey}`;
    return new WebSocket(wsUrl);
  }
}

export const apiClient = new MetronisAPI();
