import React, { useState, useRef } from 'react';

const PromptEditor = ({ image, onImageUpdate }) => {
  const [prompt, setPrompt] = useState('');
  const [styleImage, setStyleImage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [aiAnalysis, setAiAnalysis] = useState('');
  const [showAnalysis, setShowAnalysis] = useState(false);
  const fileInputRef = useRef(null);

  const handleStyleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const base64 = e.target.result.split(',')[1];
        setStyleImage({
          file,
          base64,
          preview: e.target.result,
          filename: file.name
        });
      };
      reader.readAsDataURL(file);
    }
  };

  const handleProcessImage = async () => {
    if (!image || !prompt || isProcessing) return;

    setIsProcessing(true);
    setAiAnalysis('');
    
    try {
      const response = await fetch('http://127.0.0.1:8000/edit-image', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_data: image.base64,
          prompt: prompt,
          style_transfer_image: styleImage?.base64 || null
        }),
      });

      if (response.ok) {
        const result = await response.json();
        
        if (result.success) {
          const editedImage = {
            ...image,
            base64: result.edited_image,
            preview: `data:image/png;base64,${result.edited_image}`
          };

          // Prefer backend-returned analysis; fallback to local analyze-colors if missing
          if (result.dominant_colors && typeof result.srgb_score !== 'undefined') {
            onImageUpdate(editedImage, {
              dominant_colors: result.dominant_colors,
              srgb_score: result.srgb_score,
            });
          } else {
            try {
              const colorResponse = await fetch('http://127.0.0.1:8000/analyze-colors', {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image_data: result.edited_image }),
              });
              if (colorResponse.ok) {
                const colorData = await colorResponse.json();
                if (colorData.success) {
                  onImageUpdate(editedImage, {
                    dominant_colors: colorData.dominant_colors,
                    srgb_score: colorData.srgb_score,
                  });
                } else {
                  onImageUpdate(editedImage);
                }
              } else {
                onImageUpdate(editedImage);
              }
            } catch (colorError) {
              console.error('Error analyzing colors:', colorError);
              onImageUpdate(editedImage);
            }
          }
          
          setAiAnalysis(result.message || 'AI processing completed successfully.');
          setShowAnalysis(true);
        } else {
          throw new Error('Edit processing failed');
        }
      } else {
        throw new Error('Server error: ' + response.status);
      }
    } catch (error) {
      console.error('Error processing image:', error);
      alert('Error processing edit. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const clearStyleImage = () => {
    setStyleImage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
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
      <h3 style={{ marginBottom: '15px', color: '#333' }}>AI Prompt Editor</h3>
      
      <div style={{ marginBottom: '15px' }}>
        <label style={{ 
          display: 'block', 
          marginBottom: '5px', 
          fontSize: '14px', 
          color: '#666' 
        }}>
          Describe what you want to edit:
        </label>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="e.g., 'Make the image warmer and golden', 'Add vintage style', 'Increase contrast and drama'..."
          style={{
            width: '100%',
            height: '80px',
            padding: '10px',
            border: '1px solid #ddd',
            borderRadius: '4px',
            fontSize: '14px',
            resize: 'vertical'
          }}
        />
      </div>

      <div style={{ marginBottom: '15px' }}>
        <label style={{ 
          display: 'block', 
          marginBottom: '5px', 
          fontSize: '14px', 
          color: '#666' 
        }}>
          Style Reference (Optional):
        </label>
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleStyleImageUpload}
          accept="image/*"
          style={{
            width: '100%',
            padding: '8px',
            border: '1px solid #ddd',
            borderRadius: '4px',
            fontSize: '14px'
          }}
        />
        {styleImage && (
          <div style={{ 
            marginTop: '10px', 
            padding: '10px', 
            backgroundColor: '#f9f9f9',
            borderRadius: '4px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <img 
                src={styleImage.preview} 
                alt="Style reference" 
                style={{ 
                  width: '40px', 
                  height: '40px', 
                  objectFit: 'cover',
                  borderRadius: '4px'
                }}
              />
              <span style={{ fontSize: '12px', color: '#666' }}>
                {styleImage.filename}
              </span>
            </div>
            <button
              onClick={clearStyleImage}
              style={{
                padding: '4px 8px',
                backgroundColor: '#ff6b6b',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                fontSize: '12px',
                cursor: 'pointer'
              }}
            >
              Remove
            </button>
          </div>
        )}
      </div>

      <button
        onClick={handleProcessImage}
        disabled={!image || !prompt || isProcessing}
        style={{
          width: '100%',
          padding: '12px',
          backgroundColor: isProcessing ? '#ccc' : '#007bff',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          fontSize: '14px',
          cursor: isProcessing ? 'not-allowed' : 'pointer',
          marginBottom: '15px'
        }}
      >
        {isProcessing ? 'Processing with AI...' : 'Edit with AI'}
      </button>

      {aiAnalysis && (
        <div style={{ marginTop: '15px' }}>
          <button
            onClick={() => setShowAnalysis(!showAnalysis)}
            style={{
              width: '100%',
              padding: '8px',
              backgroundColor: '#f8f9fa',
              border: '1px solid #ddd',
              borderRadius: '4px',
              fontSize: '12px',
              cursor: 'pointer',
              marginBottom: showAnalysis ? '10px' : '0'
            }}
          >
            {showAnalysis ? 'Hide' : 'Show'} AI Analysis
          </button>
          
          {showAnalysis && (
            <div style={{
              padding: '10px',
              backgroundColor: '#f8f9fa',
              border: '1px solid #e9ecef',
              borderRadius: '4px',
              fontSize: '12px',
              color: '#495057',
              maxHeight: '200px',
              overflowY: 'auto'
            }}>
              <pre style={{ 
                whiteSpace: 'pre-wrap', 
                margin: 0,
                fontFamily: 'inherit'
              }}>
                {aiAnalysis}
              </pre>
            </div>
          )}
        </div>
      )}

      <div style={{ 
        fontSize: '11px', 
        color: '#999', 
        textAlign: 'center',
        marginTop: '10px'
      }}>
        
      </div>
    </div>
  );
};

export default PromptEditor;
