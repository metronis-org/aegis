/**
 * Analytics - Charts and metrics
 */

import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { apiClient } from '../api/client';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

const Analytics: React.FC = () => {
  const { data: usage } = useQuery({
    queryKey: ['usage'],
    queryFn: () => apiClient.getUsageSummary(),
  });

  const { data: traces } = useQuery({
    queryKey: ['traces'],
    queryFn: () => apiClient.listTraces({ limit: 100 }),
  });

  // Mock data for charts
  const domainData = [
    { name: 'Healthcare', value: 45 },
    { name: 'Trading', value: 30 },
    { name: 'Robotics', value: 15 },
    { name: 'Legal', value: 10 },
  ];

  const timeSeriesData = [
    { date: '2025-10-24', traces: 20 },
    { date: '2025-10-25', traces: 35 },
    { date: '2025-10-26', traces: 28 },
    { date: '2025-10-27', traces: 42 },
    { date: '2025-10-28', traces: 38 },
    { date: '2025-10-29', traces: 50 },
    { date: '2025-10-30', traces: 45 },
  ];

  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">Analytics</h1>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <StatCard title="Total Traces" value={traces?.length || 0} />
        <StatCard title="Total Cost" value={`$${usage?.total_cost?.toFixed(2) || '0.00'}`} />
        <StatCard title="Avg Response Time" value="245ms" />
        <StatCard title="Success Rate" value="98.5%" />
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Traces Over Time */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold mb-4">Traces Over Time</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={timeSeriesData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="traces" stroke="#8884d8" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Traces by Domain */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold mb-4">Traces by Domain</h2>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={domainData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={(entry) => entry.name}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {domainData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Usage by Metric Type */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold mb-4">Usage by Type</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={Object.entries(usage?.metrics || {}).map(([key, val]: any) => ({ name: key, count: val.count }))}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="count" fill="#82ca9d" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Cost Breakdown */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold mb-4">Cost Breakdown</h2>
          <div className="space-y-4">
            {usage?.metrics && Object.entries(usage.metrics).map(([key, val]: any) => (
              <div key={key} className="flex justify-between items-center">
                <span className="text-sm text-gray-600">{key}</span>
                <div className="text-right">
                  <div className="text-sm font-semibold">${val.cost.toFixed(2)}</div>
                  <div className="text-xs text-gray-500">{val.count} units</div>
                </div>
              </div>
            ))}
          </div>
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

export default Analytics;
