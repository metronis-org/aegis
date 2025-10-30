/**
 * Expert Review - Active learning labeling interface
 */

import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

interface ReviewTrace {
  trace_id: string;
  model: string;
  domain: string;
  input_text: string;
  output_text: string;
  created_at: string;
}

const ExpertReview: React.FC = () => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [label, setLabel] = useState<'pass' | 'fail' | 'needs_review'>('pass');
  const [confidence, setConfidence] = useState(0.8);
  const [notes, setNotes] = useState('');

  const queryClient = useQueryClient();

  const { data: queueData, isLoading } = useQuery({
    queryKey: ['review-queue'],
    queryFn: async () => {
      const response = await axios.get(`${API_BASE_URL}/api/v1/expert-review/queue`, {
        headers: { Authorization: `Bearer ${localStorage.getItem('api_key')}` },
      });
      return response.data;
    },
  });

  const submitLabelMutation = useMutation({
    mutationFn: async (data: any) => {
      await axios.post(`${API_BASE_URL}/api/v1/expert-review/label`, data, {
        headers: { Authorization: `Bearer ${localStorage.getItem('api_key')}` },
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['review-queue'] });
      setCurrentIndex((prev) => prev + 1);
      setNotes('');
    },
  });

  const queue: ReviewTrace[] = queueData?.queue || [];
  const currentTrace = queue[currentIndex];

  const handleSubmit = () => {
    if (!currentTrace) return;

    submitLabelMutation.mutate({
      trace_id: currentTrace.trace_id,
      expert_email: 'expert@example.com',  // TODO: Get from auth
      label,
      confidence,
      notes: notes || null,
    });
  };

  const handleSkip = () => {
    setCurrentIndex((prev) => Math.min(prev + 1, queue.length - 1));
  };

  if (isLoading) {
    return <div className="p-6">Loading review queue...</div>;
  }

  if (!currentTrace) {
    return (
      <div className="p-6">
        <h1 className="text-3xl font-bold mb-6">Expert Review</h1>
        <div className="bg-green-50 border border-green-200 rounded-lg p-6 text-center">
          <p className="text-green-800 text-lg">All traces have been reviewed!</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">Expert Review</h1>

      {/* Progress */}
      <div className="bg-white rounded-lg shadow p-4 mb-6">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm text-gray-600">
            Reviewing {currentIndex + 1} of {queue.length}
          </span>
          <span className="text-sm text-gray-600">
            {Math.round(((currentIndex + 1) / queue.length) * 100)}% complete
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-blue-600 h-2 rounded-full"
            style={{ width: `${((currentIndex + 1) / queue.length) * 100}%` }}
          />
        </div>
      </div>

      {/* Trace Details */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Input */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold mb-4">Input</h2>
          <div className="bg-gray-50 rounded p-4 max-h-96 overflow-y-auto">
            <p className="text-sm whitespace-pre-wrap">{currentTrace.input_text}</p>
          </div>
          <div className="mt-4 text-sm text-gray-600">
            <p>Model: {currentTrace.model}</p>
            <p>Domain: {currentTrace.domain}</p>
          </div>
        </div>

        {/* Output */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold mb-4">Output</h2>
          <div className="bg-gray-50 rounded p-4 max-h-96 overflow-y-auto">
            <p className="text-sm whitespace-pre-wrap">{currentTrace.output_text}</p>
          </div>
        </div>
      </div>

      {/* Labeling Form */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-lg font-semibold mb-4">Your Review</h2>

        {/* Label Selection */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">Label</label>
          <div className="flex gap-4">
            <button
              className={`px-6 py-2 rounded ${
                label === 'pass'
                  ? 'bg-green-600 text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
              onClick={() => setLabel('pass')}
            >
              Pass
            </button>
            <button
              className={`px-6 py-2 rounded ${
                label === 'fail'
                  ? 'bg-red-600 text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
              onClick={() => setLabel('fail')}
            >
              Fail
            </button>
            <button
              className={`px-6 py-2 rounded ${
                label === 'needs_review'
                  ? 'bg-yellow-600 text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
              onClick={() => setLabel('needs_review')}
            >
              Needs Review
            </button>
          </div>
        </div>

        {/* Confidence */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Confidence: {confidence.toFixed(2)}
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={confidence}
            onChange={(e) => setConfidence(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>

        {/* Notes */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">Notes (optional)</label>
          <textarea
            className="w-full border rounded px-3 py-2"
            rows={3}
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Add any notes about this trace..."
          />
        </div>

        {/* Actions */}
        <div className="flex gap-4">
          <button
            onClick={handleSubmit}
            className="px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            disabled={submitLabelMutation.isPending}
          >
            {submitLabelMutation.isPending ? 'Submitting...' : 'Submit & Next'}
          </button>
          <button
            onClick={handleSkip}
            className="px-6 py-2 bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
          >
            Skip
          </button>
        </div>
      </div>
    </div>
  );
};

export default ExpertReview;
