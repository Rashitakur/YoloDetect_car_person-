// pages/index.js
import React, { useState } from 'react';
import axios from 'axios';

export default function Home() {
  const [file, setFile] = useState(null);
  const [response, setResponse] = useState(null);

  const handleUpload = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("image", file);

    try {
      const res = await axios.post("https://your-flask-backend.onrender.com/predict", formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResponse(res.data.detections);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div style={{ padding: '2rem' }}>
      <h1>🚗 YOLOv8 Image Detector</h1>
      <input type="file" accept="image/*" onChange={(e) => setFile(e.target.files[0])} />
      <button onClick={handleUpload} style={{ marginTop: '10px' }}>Detect</button>
      {response && (
        <div>
          <h2>Detections:</h2>
          <ul>
            {response.map((item, idx) => <li key={idx}>{item}</li>)}
          </ul>
        </div>
      )}
    </div>
  );
}
