/**
 * Dashboard Component - Main overview page
 */

import React, { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../api/client';

export const Dashboard: React.FC = () => {
  const [stats, setStats] = useState({
    total_traces: 0,
    pass_rate: 0,
    avg_execution_time: 0,
    total_cost: 0,
  });

  const { data: traces } = useQuery({
    queryKey: ['traces'],
    queryFn: () => apiClient.listTraces({ limit: 100 }),
  });

  const { data: evaluations } = useQuery({
    queryKey: ['evaluations'],
    queryFn: () => apiClient.listEvaluations({ limit: 100 }),
  });

  const { data: usage } = useQuery({
    queryKey: ['usage'],
    queryFn: () => apiClient.getUsageSummary(),
  });

  useEffect(() => {
    if (traces && evaluations) {
      const passedCount = evaluations.filter((e) => e.overall_passed).length;
      const passRate = evaluations.length > 0 ? (passedCount / evaluations.length) * 100 : 0;
      const avgTime =
        evaluations.reduce((sum, e) => sum + e.execution_time_ms, 0) / evaluations.length || 0;

      setStats({
        total_traces: traces.length,
        pass_rate: passRate,
        avg_execution_time: avgTime,
        total_cost: usage?.total_cost || 0,
      });
    }
  }, [traces, evaluations, usage]);

  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">Dashboard</h1>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <StatCard title="Total Traces" value={stats.total_traces} />
        <StatCard title="Pass Rate" value={`${stats.pass_rate.toFixed(1)}%`} />
        <StatCard title="Avg Execution Time" value={`${stats.avg_execution_time.toFixed(0)}ms`} />
        <StatCard title="Total Cost" value={`$${stats.total_cost.toFixed(2)}`} />
      </div>

      {/* Recent Traces */}
      <div className="bg-white rounded-lg shadow p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4">Recent Traces</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Trace ID
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Model
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Domain
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Created
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {traces?.slice(0, 10).map((trace) => (
                <tr key={trace.trace_id}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    {trace.trace_id.substring(0, 8)}...
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {trace.model}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {trace.domain}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {new Date(trace.created_at).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

interface StatCardProps {
  title: string;
  value: string | number;
}

const StatCard: React.FC<StatCardProps> = ({ title, value }) => (
  <div className="bg-white rounded-lg shadow p-6">
    <p className="text-sm font-medium text-gray-600 mb-2">{title}</p>
    <p className="text-2xl font-bold text-gray-900">{value}</p>
  </div>
);
