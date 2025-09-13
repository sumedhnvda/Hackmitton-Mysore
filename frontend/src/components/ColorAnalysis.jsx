import React from 'react';

const ColorAnalysis = ({ analysis }) => {
  if (!analysis) {
    return (
      <div style={{
        backgroundColor: 'white',
        border: '1px solid #ddd',
        borderRadius: '8px',
        padding: '20px',
        marginBottom: '20px'
      }}>
        <h3 style={{ marginBottom: '15px', color: '#333' }}>Color Analysis</h3>
        <p style={{ color: '#666', fontSize: '14px' }}>Upload an image to see color analysis</p>
      </div>
    );
  }

  return (
    <div style={{
      backgroundColor: 'white',
      border: '1px solid #ddd',
      borderRadius: '8px',
      padding: '20px',
      marginBottom: '20px'
    }}>
      <h3 style={{ marginBottom: '15px', color: '#333' }}>Color Analysis</h3>
      
      <div style={{ marginBottom: '15px' }}>
        <h4 style={{ fontSize: '16px', marginBottom: '8px', color: '#444' }}>Color Richness Score</h4>
        <div style={{ 
          fontSize: '24px', 
          fontWeight: 'bold', 
          color: getScoreColor(analysis.srgb_score),
          padding: '10px',
          backgroundColor: '#f8f9fa',
          borderRadius: '4px',
          textAlign: 'center'
        }}>
          {analysis.srgb_score}%
        </div>
        <p style={{ fontSize: '12px', color: '#666', marginTop: '4px', textAlign: 'center' }}>
          Higher score indicates better color distribution and variety
        </p>
      </div>

      {analysis.dominant_colors && analysis.dominant_colors.length > 0 && (
        <div>
          <h4 style={{ fontSize: '16px', marginBottom: '10px', color: '#444' }}>Dominant Colors</h4>
          <div style={{ display: 'grid', gap: '8px' }}>
            {analysis.dominant_colors.map((color, index) => (
              <div key={index} style={{
                display: 'flex',
                alignItems: 'center',
                padding: '8px',
                backgroundColor: '#f8f9fa',
                borderRadius: '4px'
              }}>
                <div style={{
                  width: '30px',
                  height: '30px',
                  backgroundColor: color.hex,
                  borderRadius: '4px',
                  marginRight: '10px',
                  border: '1px solid #ddd'
                }}></div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: '12px', fontWeight: 'bold', color: '#333' }}>
                    {color.hex}
                  </div>
                  <div style={{ fontSize: '11px', color: '#666' }}>
                    RGB({color.rgb[0]}, {color.rgb[1]}, {color.rgb[2]})
                  </div>
                </div>
                <div style={{ fontSize: '12px', color: '#666', fontWeight: 'bold' }}>
                  {color.percentage}%
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// Helper function to determine score color based on value
const getScoreColor = (score) => {
  if (score >= 80) return '#2ecc71'; // Green for high scores
  if (score >= 60) return '#3498db'; // Blue for good scores
  if (score >= 40) return '#f39c12'; // Orange for medium scores
  return '#e74c3c'; // Red for low scores
};

export default ColorAnalysis;