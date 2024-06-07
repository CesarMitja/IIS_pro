const axios = require('axios');

const fetchMetrics = async () => {
  try {
    const response = await axios.get('http://localhost:5000/metrics');
    console.log('Metrics fetched:', response.data);
  } catch (error) {
    console.error('Error fetching metrics:', error);
  }
};

// Function to calculate milliseconds until next midnight
const msUntilMidnight = () => {
  const now = new Date();
  const midnight = new Date(now);
  midnight.setHours(24, 0, 0, 0);
  return midnight - now;
};

// Function to schedule the task for every day
const scheduleDailyTask = () => {
  setTimeout(() => {
    fetchMetrics();
    scheduleDailyTask();
  }, msUntilMidnight());
};

// Start the scheduler
scheduleDailyTask();
