import React, { useState } from 'react';
import axios from 'axios';
import './App.css';
import Form from './Form';
import House3D from './House3D';
import CanvasArea from './CanvasArea';
import AdminDashboard from './AdminDashboard';

function App() {
  const [currentPage, setCurrentPage] = useState('rent');
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  const [rentFormData, setRentFormData] = useState({
    Bedrooms: '',
    Bathrooms: '',
    LivingArea: '',
    Type: ''
  });

  const [priceFormData, setPriceFormData] = useState({
    Bedrooms: '',
    Bathrooms: '',
    LivingArea: '',
    LotArea: '',
    Type: ''
  });

  const [rentPrediction, setRentPrediction] = useState(null);
  const [pricePrediction, setPricePrediction] = useState(null);
  const [houseParams, setHouseParams] = useState({
    bedrooms: 3,
    bathrooms: 2,
    livingArea: 1600,
    lotArea: 7000,
  });

  const handleRentChange = (e) => {
    const { name, value } = e.target;
    setRentFormData({
      ...rentFormData,
      [name]: value,
    });
  };

  const handlePriceChange = (e) => {
    const { name, value } = e.target;
    setPriceFormData({
      ...priceFormData,
      [name]: value,
    });
  };

  const handleRentSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:5000/predict_rent', {
        Bedrooms: parseInt(rentFormData.Bedrooms),
        Bathrooms: parseInt(rentFormData.Bathrooms),
        'Living Area': parseInt(rentFormData.LivingArea),
        Type: rentFormData.Type
      });
      setRentPrediction(response.data.prediction);
    } catch (error) {
      console.error('Error making rent prediction', error);
    }
  };

  const handlePriceSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:5000/predict_price', {
        Bedrooms: parseInt(priceFormData.Bedrooms),
        Bathrooms: parseInt(priceFormData.Bathrooms),
        'Living Area': parseInt(priceFormData.LivingArea),
        'Lot Area': parseInt(priceFormData.LotArea),
        Type: priceFormData.Type
      });
      setPricePrediction(response.data.prediction);
    } catch (error) {
      console.error('Error making price prediction', error);
    }
  };

  const updateLivingArea = (area) => {
    setHouseParams(prevState => ({ ...prevState, livingArea: area }));
    setPriceFormData(prevState => ({ ...prevState, LivingArea: area }));
  };

  const updateLotArea = (area) => {
    setHouseParams(prevState => ({ ...prevState, lotArea: area }));
    setPriceFormData(prevState => ({ ...prevState, LotArea: area }));
  };

  const handleAdminAccess = () => {
    const password = prompt('Please enter the admin password:');
    if (password === 'admin') { // Replace 'your_admin_password' with your actual password
      setIsAuthenticated(true);
      setCurrentPage('admin');
    } else {
      alert('Incorrect password');
    }
  };

  return (
    <div className="App-container">
      <nav className="App-nav">
        <button onClick={() => setCurrentPage('rent')}>Rent Prediction</button>
        <button onClick={() => setCurrentPage('price')}>Price Prediction</button>
        <button onClick={handleAdminAccess}>Admin Access</button>
      </nav>
      <div className="App-content">
        {currentPage === 'rent' ? (
          <div className="form-container">
            <div className="App-form">
              <h2>Rent Prediction</h2>
              <form onSubmit={handleRentSubmit}>
                <div>
                  <label>Bedrooms:</label>
                  <input type="number" name="Bedrooms" value={rentFormData.Bedrooms} onChange={handleRentChange} required />
                </div>
                <div>
                  <label>Bathrooms:</label>
                  <input type="number" name="Bathrooms" value={rentFormData.Bathrooms} onChange={handleRentChange} required />
                </div>
                <div>
                  <label>Living Area (sq ft):</label>
                  <input type="number" name="LivingArea" value={rentFormData.LivingArea} onChange={handleRentChange} required />
                </div>
                <div>
                  <label>Type:</label>
                  <select name="Type" value={rentFormData.Type} onChange={handleRentChange} required>
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
              {rentPrediction && (
                <div>
                  <h2>Predicted Rent:</h2>
                  <p>{rentPrediction}</p>
                </div>
              )}
            </div>
            <div className="App-3dmodel">
              <House3D {...houseParams} />
            </div>
          </div>
        ) : currentPage === 'price' ? (
          <div className="form-container">
            <div className="App-form">
              <h2>Price Prediction</h2>
              <form onSubmit={handlePriceSubmit}>
                <div>
                  <label>Bedrooms:</label>
                  <input type="number" name="Bedrooms" value={priceFormData.Bedrooms} onChange={handlePriceChange} required />
                </div>
                <div>
                  <label>Bathrooms:</label>
                  <input type="number" name="Bathrooms" value={priceFormData.Bathrooms} onChange={handlePriceChange} required />
                </div>
                <div>
                  <label>Living Area (sq ft):</label>
                  <input type="number" name="LivingArea" value={priceFormData.LivingArea} onChange={handlePriceChange} required />
                </div>
                <div>
                  <label>Lot Area (sq ft):</label>
                  <input type="number" name="LotArea" value={priceFormData.LotArea} onChange={handlePriceChange} required />
                </div>
                <div>
                  <label>Type:</label>
                  <select name="Type" value={priceFormData.Type} onChange={handlePriceChange} required>
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
              {pricePrediction && (
                <div>
                  <h2>Predicted Price:</h2>
                  <p>{pricePrediction}</p>
                </div>
              )}
            </div>
            <div className="App-3dmodel">
              <House3D {...houseParams} />
            </div>
            {/* <div className="App-canvas">
              <CanvasArea
                updateLivingArea={updateLivingArea}
                updateLotArea={updateLotArea}
                initialLivingArea={houseParams.livingArea}
                initialLotArea={houseParams.lotArea}
              />
            </div> */}
          </div>
        ) : isAuthenticated ? (
          <AdminDashboard />
        ) : (
          <div>
            <h2>Access Denied</h2>
            <p>You must enter the correct password to access the admin dashboard.</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
