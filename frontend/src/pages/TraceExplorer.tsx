/**
 * Trace Explorer - Browse and search all traces
 */

import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../api/client';

export const TraceExplorer: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [domainFilter, setDomainFilter] = useState<string>('');

  const { data: traces, isLoading } = useQuery({
    queryKey: ['traces', domainFilter],
    queryFn: () => apiClient.listTraces({ domain: domainFilter || undefined, limit: 100 }),
  });

  const filteredTraces = traces?.filter((trace) =>
    searchQuery
      ? trace.input.toLowerCase().includes(searchQuery.toLowerCase()) ||
        trace.output.toLowerCase().includes(searchQuery.toLowerCase())
      : true
  );

  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">Trace Explorer</h1>

      {/* Search and Filters */}
      <div className="bg-white rounded-lg shadow p-4 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <input
            type="text"
            placeholder="Search traces..."
            className="border rounded px-4 py-2"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          <select
            className="border rounded px-4 py-2"
            value={domainFilter}
            onChange={(e) => setDomainFilter(e.target.value)}
          >
            <option value="">All Domains</option>
            <option value="healthcare">Healthcare</option>
            <option value="trading">Trading</option>
            <option value="robotics">Robotics</option>
            <option value="legal">Legal</option>
          </select>
        </div>
      </div>

      {/* Traces Table */}
      <div className="bg-white rounded-lg shadow">
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
                Input Preview
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Created
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {isLoading ? (
              <tr>
                <td colSpan={6} className="px-6 py-4 text-center text-gray-500">
                  Loading...
                </td>
              </tr>
            ) : filteredTraces && filteredTraces.length > 0 ? (
              filteredTraces.map((trace) => (
                <tr key={trace.trace_id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-mono">
                    {trace.trace_id.substring(0, 8)}...
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm">{trace.model}</td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="px-2 py-1 text-xs font-semibold rounded-full bg-blue-100 text-blue-800">
                      {trace.domain}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    {trace.input.substring(0, 50)}...
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {new Date(trace.created_at).toLocaleString()}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm">
                    <button className="text-blue-600 hover:text-blue-800">View</button>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={6} className="px-6 py-4 text-center text-gray-500">
                  No traces found
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
