import React, { useState } from 'react';

const EnhancePanel = ({ image, onImageUpdate }) => {
  const [isEnhancing, setIsEnhancing] = useState(false);

  const handleEnhance = async (scaleFactor) => {
    if (!image || isEnhancing) return;

    setIsEnhancing(true);
    try {
      const response = await fetch('http://127.0.0.1:8000/enhance-image', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_data: image.base64,
          scale_factor: scaleFactor
        }),
      });

      if (response.ok) {
        const result = await response.json();
        if (result.success) {
          // Update the image
          const enhancedImage = {
            ...image,
            base64: result.enhanced_image,
            preview: `data:image/png;base64,${result.enhanced_image}`
          };
          
          // Pass the color analysis data along with the image update
          onImageUpdate(enhancedImage, {
            dominant_colors: result.dominant_colors,
            srgb_score: result.srgb_score
          });
        }
      }
    } catch (error) {
      console.error('Error enhancing image:', error);
    } finally {
      setIsEnhancing(false);
    }
  };

  return (
    <div style={{
      backgroundColor: 'white',
      border: '1px solid #ddd',
      borderRadius: '8px',
      padding: '20px',
      marginBottom: '20px'
    }}>
      <h3 style={{ marginBottom: '15px', color: '#333' }}>Scaling Factor</h3>
      
      <div style={{ display: 'grid', gap: '10px' }}>
        <button
          onClick={() => handleEnhance(2)}
          disabled={!image || isEnhancing}
          style={{
            padding: '10px',
            backgroundColor: '#28a745',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '14px'
          }}
        >
          2x 
        </button>
        
        <button
          onClick={() => handleEnhance(4)}
          disabled={!image || isEnhancing}
          style={{
            padding: '10px',
            backgroundColor: '#17a2b8',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '14px'
          }}
        >
          4x 
        </button>
        
        <button
          onClick={() => handleEnhance(8)}
          disabled={!image || isEnhancing}
          style={{
            padding: '10px',
            backgroundColor: '#6f42c1',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '14px'
          }}
        >
          8x 
        </button>
      </div>
      
      {isEnhancing && (
        <p style={{ marginTop: '10px', fontSize: '14px', color: '#666' }}>
          Enhancing image...
        </p>
      )}
    </div>
  );
};

export default EnhancePanel;
