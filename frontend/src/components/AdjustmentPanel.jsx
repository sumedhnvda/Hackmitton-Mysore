import React, { useState, useEffect } from 'react';

const AdjustmentPanel = ({ image, onImageUpdate }) => {
  const [adjustments, setAdjustments] = useState({
    brightness: 0,
    contrast: 0,
    saturation: 0,
    exposure: 0,
    shadows: 0,
    highlights: 0,
  });

  // Reset adjustments when image changes externally (like reset button)
  useEffect(() => {
    if (image && image.adjustments) {
      setAdjustments(image.adjustments);
    } else if (image && !image.adjustments) {
      // If image doesn't have adjustments, reset to default
      const defaultAdjustments = {
        brightness: 0,
        contrast: 0,
        saturation: 0,
        exposure: 0,
        shadows: 0,
        highlights: 0,
      };
      setAdjustments(defaultAdjustments);
    }
  }, [image]);

  const handleAdjustmentChange = (key, value) => {
    const newAdjustments = {
      ...adjustments,
      [key]: value
    };
    setAdjustments(newAdjustments);
    
    // Apply adjustments in real-time by updating the image with CSS filters
    if (image && onImageUpdate) {
      const adjustedImage = {
        ...image,
        adjustments: newAdjustments,
        cssFilter: generateCSSFilter(newAdjustments)
      };
      onImageUpdate(adjustedImage);
    }
  };

  const generateCSSFilter = (adj) => {
    const filters = [];
    
    if (adj.brightness !== 0) {
      filters.push(`brightness(${1 + adj.brightness / 100})`);
    }
    if (adj.contrast !== 0) {
      filters.push(`contrast(${1 + adj.contrast / 100})`);
    }
    if (adj.saturation !== 0) {
      filters.push(`saturate(${1 + adj.saturation / 100})`);
    }
    if (adj.exposure !== 0) {
      // Exposure can be simulated with brightness + contrast
      const exposureEffect = 1 + adj.exposure / 200;
      filters.push(`brightness(${exposureEffect})`);
      filters.push(`contrast(${1 + adj.exposure / 300})`);
    }
    if (adj.shadows !== 0) {
      // Shadows can be simulated with a subtle brightness adjustment
      filters.push(`brightness(${1 + adj.shadows / 300})`);
    }
    if (adj.highlights !== 0) {
      // Highlights can be simulated with contrast
      filters.push(`contrast(${1 + adj.highlights / 200})`);
    }
    
    return filters.length > 0 ? filters.join(' ') : 'none';
  };

  const resetAll = () => {
    const resetAdjustments = {
      brightness: 0,
      contrast: 0,
      saturation: 0,
      exposure: 0,
      shadows: 0,
      highlights: 0,
    };
    setAdjustments(resetAdjustments);
    
    if (image && onImageUpdate) {
      const resetImage = {
        ...image,
        adjustments: resetAdjustments,
        cssFilter: 'none'
      };
      onImageUpdate(resetImage);
    }
  };

  const applyToCanvas = async () => {
    if (!image) return;
    
    // Create a canvas to apply the filters permanently
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    img.onload = async () => {
      canvas.width = img.width;
      canvas.height = img.height;
      
      // Apply CSS filter to canvas context
      ctx.filter = generateCSSFilter(adjustments);
      ctx.drawImage(img, 0, 0);
      
      // Convert back to base64
      const newBase64 = canvas.toDataURL('image/png').split(',')[1];
      const permanentImage = {
        ...image,
        base64: newBase64,
        preview: canvas.toDataURL('image/png'),
        adjustments: { brightness: 0, contrast: 0, saturation: 0, exposure: 0, shadows: 0, highlights: 0 },
        cssFilter: 'none'
      };
      
      // Reset adjustments
      setAdjustments({ brightness: 0, contrast: 0, saturation: 0, exposure: 0, shadows: 0, highlights: 0 });
      
      try {
        // Perform color analysis on the adjusted image
        const response = await fetch('http://127.0.0.1:8000/analyze-colors', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ image_data: newBase64 }),
        });
        
        if (response.ok) {
          const data = await response.json();
          if (data.success) {
            // Update with color analysis data
            onImageUpdate(permanentImage, {
              dominant_colors: data.dominant_colors,
              srgb_score: data.srgb_score
            });
            return;
          }
        }
      } catch (error) {
        console.error('Error analyzing colors:', error);
      }
      
      // If color analysis failed, just update the image
      onImageUpdate(permanentImage);
    };
    
    img.src = image.preview;
  };

  const adjustmentItems = [
    { key: 'brightness', label: 'Brightness', min: -100, max: 100 },
    { key: 'contrast', label: 'Contrast', min: -100, max: 100 },
    { key: 'saturation', label: 'Saturation', min: -100, max: 100 },
    { key: 'exposure', label: 'Exposure', min: -100, max: 100 },
    { key: 'shadows', label: 'Shadows', min: -100, max: 100 },
    { key: 'highlights', label: 'Highlights', min: -100, max: 100 },
  ];

  return (
    <div className="adjustment-panel" style={{
      backgroundColor: 'white',
      border: '1px solid #ddd',
      borderRadius: '8px',
      padding: '20px',
      marginBottom: '20px'
    }}>
      <h3 style={{ marginBottom: '20px', color: '#333' }}>Image Adjustments</h3>
      <p style={{ fontSize: '12px', color: '#666', marginBottom: '15px' }}>
        Real-time adjustments - changes are applied instantly!
      </p>
      
      {adjustmentItems.map(({ key, label, min, max }) => (
        <div key={key} className="adjustment-item" style={{ marginBottom: '15px' }}>
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            marginBottom: '5px'
          }}>
            <label style={{ fontSize: '14px', color: '#666' }}>{label}</label>
            <span style={{ fontSize: '14px', color: '#999' }}>
              {adjustments[key]}%
            </span>
          </div>
          <input
            type="range"
            min={min}
            max={max}
            value={adjustments[key]}
            onChange={(e) => handleAdjustmentChange(key, parseInt(e.target.value))}
            style={{
              width: '100%',
              height: '4px',
              borderRadius: '2px',
              background: '#ddd',
              outline: 'none',
              cursor: 'pointer'
            }}
          />
        </div>
      ))}
      
      <div className="panel-controls" style={{ 
        display: 'flex', 
        gap: '10px', 
        marginTop: '20px' 
      }}>
        <button
          onClick={applyToCanvas}
          disabled={!image}
          style={{
            flex: 1,
            padding: '10px',
            backgroundColor: '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '14px'
          }}
        >
          Apply Permanently
        </button>
        
        <button
          onClick={resetAll}
          style={{
            flex: 1,
            padding: '10px',
            backgroundColor: '#6c757d',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '14px'
          }}
        >
          Reset All
        </button>
      </div>
    </div>
  );
};

export default AdjustmentPanel;
