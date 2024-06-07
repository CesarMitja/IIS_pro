const axios = require('axios');

const fetchMetrics = async () => {
  try {
    const response = await axios.get('http://localhost:5000/metrics');
    console.log('Metrics fetched:', response.data);
  } catch (error) {
    console.error('Error fetching metrics:', error);
  }
};

const msUntilMidnight = () => {
  const now = new Date();
  const midnight = new Date(now);
  midnight.setHours(24, 0, 0, 0);
  return midnight - now;
};

const scheduleDailyTask = () => {
  setTimeout(() => {
    fetchMetrics();
    scheduleDailyTask();
  }, msUntilMidnight());
};

scheduleDailyTask();
