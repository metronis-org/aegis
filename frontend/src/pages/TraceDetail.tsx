import React from 'react';
import { useParams } from 'react-router-dom';
import { Box, Typography, Paper } from '@mui/material';

const TraceDetail: React.FC = () => {
  const { traceId } = useParams<{ traceId: string }>();

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Trace Detail
      </Typography>
      <Paper sx={{ p: 3 }}>
        <Typography variant="body1">Trace ID: {traceId}</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
          Detailed trace information will be displayed here.
        </Typography>
      </Paper>
    </Box>
  );
};

export default TraceDetail;
