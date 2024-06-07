import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './AdminDashboard.css';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { Line } from 'react-chartjs-2';

// Register components with Chart.js
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

export default function AdminDashboard() {
  const [rentPredictions, setRentPredictions] = useState([]);
  const [pricePredictions, setPricePredictions] = useState([]);
  const [rentMetrics, setRentMetrics] = useState([]);
  const [priceMetrics, setPriceMetrics] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchRentPredictions();
    fetchPricePredictions();
    fetchRentMetrics();
    fetchPriceMetrics();
  }, []);

  const fetchRentPredictions = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/rent_predictions');
      setRentPredictions(response.data);
    } catch (error) {
      setError(error.message);
    }
  };

  const fetchPricePredictions = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/price_predictions');
      setPricePredictions(response.data);
    } catch (error) {
      setError(error.message);
    }
  };

  const fetchRentMetrics = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/rent_daily_metrics');
      setRentMetrics(response.data);
    } catch (error) {
      setError(error.message);
    }
  };

  const fetchPriceMetrics = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/price_daily_metrics');
      setPriceMetrics(response.data);
    } catch (error) {
      setError(error.message);
    }
  };

  const createChartData = (data, key) => {
    return {
      labels: data.map(item => new Date(item.timestamp).toLocaleDateString()),
      datasets: [{
        label: key,
        data: data.map(item => item[key]),
        borderColor: 'rgba(75,192,192,1)',
        backgroundColor: 'rgba(75,192,192,0.2)',
        fill: true,
      }]
    };
  };

  return (
    <div className="admin-dashboard">
      <h1>Admin Dashboard</h1>
      {error && <p>Error: {error}</p>}
      <div className="admin-section">
        <h2>Rent Predictions</h2>
        <table>
          <thead>
            <tr>
              <th>Bedrooms</th>
              <th>Bathrooms</th>
              <th>Living Area</th>
              <th>Type</th>
              <th>Predicted Rent</th>
              <th>Actual Rent</th>
            </tr>
          </thead>
          <tbody>
            {rentPredictions.map((prediction, index) => (
              <tr key={index}>
                <td>{prediction.Bedrooms}</td>
                <td>{prediction.Bathrooms}</td>
                <td>{prediction.Living_Area}</td>
                <td>{prediction.Type}</td>
                <td>{prediction.Prediction}</td>
                <td>{prediction.Actual}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="admin-section">
        <h2>Price Predictions</h2>
        <table>
          <thead>
            <tr>
              <th>Bedrooms</th>
              <th>Bathrooms</th>
              <th>Living Area</th>
              <th>Lot Area</th>
              <th>Type</th>
              <th>Predicted Price</th>
              <th>Actual Price</th>
            </tr>
          </thead>
          <tbody>
            {pricePredictions.map((prediction, index) => (
              <tr key={index}>
                <td>{prediction.Bedrooms}</td>
                <td>{prediction.Bathrooms}</td>
                <td>{prediction.Living_Area}</td>
                <td>{prediction.Lot_Area}</td>
                <td>{prediction.Type}</td>
                <td>{prediction.Prediction}</td>
                <td>{prediction.Actual}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="admin-section">
        <h2>Rent Metrics</h2>
        <Line data={createChartData(rentMetrics, 'Rent_MSE')} />
        <Line data={createChartData(rentMetrics, 'Rent_MAE')} />
      </div>
      <div className="admin-section">
        <h2>Price Metrics</h2>
        <Line data={createChartData(priceMetrics, 'Price_MSE')} />
        <Line data={createChartData(priceMetrics, 'Price_MAE')} />
      </div>
    </div>
  );
}
