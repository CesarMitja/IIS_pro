import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Real Estate Prediction</h1>
        <Navigation />
      </header>
    </div>
  );
}

function Navigation() {
  const [currentPage, setCurrentPage] = useState('rent');

  return (
    <div className="App-container">
      <nav className="App-nav">
        <button onClick={() => setCurrentPage('rent')}>Rent Prediction</button>
        <button onClick={() => setCurrentPage('price')}>Price Prediction</button>
      </nav>
      <div className="App-content">
        {currentPage === 'rent' && <RentPrediction />}
        {currentPage === 'price' && <PricePrediction />}
      </div>
    </div>
  );
}

function RentPrediction() {
  const [formData, setFormData] = useState({
    Bedrooms: '',
    Bathrooms: '',
    LivingArea: '',
    Type: ''
  });

  const [prediction, setPrediction] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:5000/predict_rent', {
        Bedrooms: parseInt(formData.Bedrooms),
        Bathrooms: parseInt(formData.Bathrooms),
        'Living Area': parseInt(formData.LivingArea),
        Type: formData.Type
      });
      setPrediction(response.data.prediction);
    } catch (error) {
      console.error('Error making rent prediction', error);
    }
  };

  return (
    <div>
      <h2>Rent Prediction</h2>
      <form onSubmit={handleSubmit}>
        <div>
          <label>Bedrooms:</label>
          <input type="number" name="Bedrooms" value={formData.Bedrooms} onChange={handleChange} required />
        </div>
        <div>
          <label>Bathrooms:</label>
          <input type="number" name="Bathrooms" value={formData.Bathrooms} onChange={handleChange} required />
        </div>
        <div>
          <label>Living Area (sq ft):</label>
          <input type="number" name="LivingArea" value={formData.LivingArea} onChange={handleChange} required />
        </div>
        <div>
          <label>Type:</label>
          <select name="Type" value={formData.Type} onChange={handleChange} required>
            <option value="">Select Type</option>
            <option value="SINGLE_FAMILY">SINGLE_FAMILY</option>
            <option value="APARTMENT">APARTMENT</option>
            <option value="MULTI_FAMILY">MULTI_FAMILY</option>
            <option value="CONDO">CONDO</option>
            <option value="TOWNHOUSE">TOWNHOUSE</option>
          </select>
        </div>
        <button type="submit">Predict Rent</button>
      </form>
      {prediction && (
        <div>
          <h2>Predicted Rent:</h2>
          <p>{prediction}</p>
        </div>
      )}
    </div>
  );
}

function PricePrediction() {
  const [formData, setFormData] = useState({
    Bedrooms: '',
    Bathrooms: '',
    LivingArea: '',
    LotArea: '',
    Type: ''
  });

  const [prediction, setPrediction] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:5000/predict_price', {
        Bedrooms: parseInt(formData.Bedrooms),
        Bathrooms: parseInt(formData.Bathrooms),
        'Living Area': parseInt(formData.LivingArea),
        'Lot Area': parseInt(formData.LotArea),
        Type: formData.Type
      });
      setPrediction(response.data.prediction);
    } catch (error) {
      console.error('Error making price prediction', error);
    }
  };

  return (
    <div>
      <h2>Price Prediction</h2>
      <form onSubmit={handleSubmit}>
        <div>
          <label>Bedrooms:</label>
          <input type="number" name="Bedrooms" value={formData.Bedrooms} onChange={handleChange} required />
        </div>
        <div>
          <label>Bathrooms:</label>
          <input type="number" name="Bathrooms" value={formData.Bathrooms} onChange={handleChange} required />
        </div>
        <div>
          <label>Living Area (sq ft):</label>
          <input type="number" name="LivingArea" value={formData.LivingArea} onChange={handleChange} required />
        </div>
        <div>
          <label>Lot Area (sq ft):</label>
          <input type="number" name="LotArea" value={formData.LotArea} onChange={handleChange} required />
        </div>
        <div>
          <label>Type:</label>
          <select name="Type" value={formData.Type} onChange={handleChange} required>
            <option value="">Select Type</option>
            <option value="SINGLE_FAMILY">SINGLE_FAMILY</option>
            <option value="APARTMENT">APARTMENT</option>
            <option value="MULTI_FAMILY">MULTI_FAMILY</option>
            <option value="CONDO">CONDO</option>
            <option value="TOWNHOUSE">TOWNHOUSE</option>
          </select>
        </div>
        <button type="submit">Predict Price</button>
      </form>
      {prediction && (
        <div>
          <h2>Predicted Price:</h2>
          <p>{prediction}</p>
        </div>
      )}
    </div>
  );
}

export default App;
