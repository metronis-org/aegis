import React from 'react';
import { Box, Typography, Paper } from '@mui/material';

const Domains: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Domains
      </Typography>
      <Paper sx={{ p: 3 }}>
        <Typography variant="body1">
          Domain configuration and management will be displayed here.
        </Typography>
      </Paper>
    </Box>
  );
};

export default Domains;
