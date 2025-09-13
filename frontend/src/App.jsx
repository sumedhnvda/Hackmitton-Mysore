import React, { useState, useRef } from 'react';
import { useTheme } from './theme';
import ImageUploader from './components/ImageUploader';
import AdjustmentPanel from './components/AdjustmentPanel';
import ColorAnalysis from './components/ColorAnalysis';
import EnhancePanel from './components/EnhancePanel';
import PromptEditor from './components/PromptEditor';
import './styles/App.css';

const App = () => {
  const { theme, toggleTheme } = useTheme();
  const [uploadedImage, setUploadedImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [originalImage, setOriginalImage] = useState(null); // Store original for reset
  const [colorAnalysis, setColorAnalysis] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const canvasRef = useRef(null);

  const handleImageUpload = (imageData) => {
    setUploadedImage(imageData);
    setProcessedImage(imageData);
    setOriginalImage(imageData); // Store original
    setColorAnalysis(null);
    
    if (imageData) {
      // Analyze colors
      analyzeColors(imageData.base64);
    }
  };

  const analyzeColors = async (base64Data) => {
  try {
      const response = await fetch('http://127.0.0.1:8000/analyze-colors', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image_data: base64Data }),
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          // Format analysis data for the component
          setColorAnalysis({
            srgb_score: data.srgb_score,
            dominant_colors: data.dominant_colors
          });
        }
      }
    } catch (error) {
      console.error('Error analyzing colors:', error);
    }
  };

  const handleColorizeImage = async () => {
    if (!uploadedImage) return;
    
    setIsProcessing(true);
    try {
      const response = await fetch('http://127.0.0.1:8002/colorize-image', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image_data: uploadedImage.base64 }),
      });
      
      if (response.ok) {
        const result = await response.json();
        const colorizedImage = {
          ...uploadedImage,
          base64: result.colorized_image,
          preview: `data:image/png;base64,${result.colorized_image}`
        };
        
        // Update image and color analysis if provided
        if (result.dominant_colors && result.srgb_score) {
          updateProcessedImage(colorizedImage, {
            dominant_colors: result.dominant_colors,
            srgb_score: result.srgb_score
          });
        } else {
          updateProcessedImage(colorizedImage);
        }
      }
    } catch (error) {
      console.error('Error colorizing image:', error);
    } finally {
      setIsProcessing(false);
    }
  };
  
  const handlePretrainedColorize = async () => {
    if (!uploadedImage) return;
    setIsProcessing(true);
    try {
      const apiKey = import.meta.env.VITE_DEEPAI_API_KEY;
      const resp = await fetch('http://127.0.0.1:8000/pretrained-colorize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_data: uploadedImage.base64,
          ...(apiKey ? { api_key: apiKey } : {}),
        }),
      });
      if (!resp.ok) throw new Error(`Backend error: ${resp.status}`);
      const result = await resp.json();
      const colorizedImage = {
        ...uploadedImage,
        base64: result.colorized_image,
        preview: `data:image/png;base64,${result.colorized_image}`,
      };
      if (result.dominant_colors && typeof result.srgb_score !== 'undefined') {
        updateProcessedImage(colorizedImage, {
          dominant_colors: result.dominant_colors,
          srgb_score: result.srgb_score,
        });
      } else {
        updateProcessedImage(colorizedImage);
      }
    } catch (error) {
      console.error('Error with pretrained colorization:', error);
      alert('Pretrained colorization failed. Ensure DEEPAI_API_KEY is set in backend/.env or pass VITE_DEEPAI_API_KEY.');
    } finally {
      setIsProcessing(false);
    }
  };

  // Function to update the processed image and optionally update color analysis
  const updateProcessedImage = (newImage, colorData = null) => {
    setProcessedImage(newImage);
    
    // If color data is provided, update the color analysis
    if (colorData) {
      setColorAnalysis(colorData);
    } else if (newImage && newImage.base64) {
      // If no color data is provided but we have a new image, analyze colors
      analyzeColors(newImage.base64);
    }
  };

  const handleReset = () => {
    if (originalImage) {
      // Reset to original image and clear any adjustments/filters
      const resetImage = {
        ...originalImage,
        adjustments: {
          brightness: 0,
          contrast: 0,
          saturation: 0,
          exposure: 0,
          shadows: 0,
          highlights: 0,
        },
        cssFilter: 'none'
      };
      updateProcessedImage(resetImage);
    }
  };

  const handleDownload = () => {
    if (!processedImage) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      
      // Generate download
      const timestamp = Date.now();
      const filename = `vinci-edited-${timestamp}.png`;
      
      canvas.toBlob((blob) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      });
    };
    
    img.src = processedImage.preview;
  };

  return (
    <div className={`app ${theme}`}>
      <header className="header">
        <h1>VINCI</h1>
        <div className="header-controls">
          <button onClick={toggleTheme} className="theme-toggle">
            {theme === 'light' ? '🌙' : '☀️'}
          </button>
        </div>
      </header>

      <main className="main-content">
        {/* Upload Section */}
        <div className="upload-section">
          <ImageUploader 
            onImageUpload={handleImageUpload}
            uploadedImage={uploadedImage}
          />
        </div>

        {uploadedImage && (
          <div className="editor-section">
            {/* Main Image Editor */}
            <div className="image-editor">
              <div className="image-display">
                {processedImage && (
                  <img 
                    src={processedImage.preview} 
                    alt="Processed" 
                    className="main-image"
                    style={{
                      filter: processedImage.cssFilter || 'none',
                      transition: 'filter 0.1s ease'
                    }}
                  />
                )}
              </div>
              
              <div className="image-controls">
                <button 
                  onClick={handleColorizeImage}
                  disabled={isProcessing}
                  className="colorize-btn"
                  style={{
                    padding: '10px 15px',
                    backgroundColor: '#007bff',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: isProcessing ? 'not-allowed' : 'pointer',
                    marginRight: '10px'
                  }}
                >
                  {isProcessing ? 'Processing...' : 'Custom Colorize'}
                </button>
                
                <button 
                  onClick={handlePretrainedColorize}
                  disabled={isProcessing}
                  className="pretrained-btn"
                  style={{
                    padding: '10px 15px',
                    backgroundColor: '#28a745',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: isProcessing ? 'not-allowed' : 'pointer',
                    marginRight: '10px'
                  }}
                >
                  {isProcessing ? 'Processing...' : 'Pretrained B&W to Colorized'}
                </button>
                
                <button 
                  onClick={handleReset}
                  disabled={!originalImage}
                  className="reset-btn"
                  style={{
                    padding: '10px 15px',
                    backgroundColor: '#6c757d',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    marginRight: '10px'
                  }}
                >
                  Reset to Original
                </button>
                
                <button 
                  onClick={handleDownload}
                  disabled={!processedImage}
                  className="download-btn"
                  style={{
                    padding: '10px 15px',
                    backgroundColor: '#17a2b8',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer'
                  }}
                >
                  Download Image
                </button>
              </div>
            </div>

            {/* Sidebar with controls */}
            <div className="sidebar">
              <AdjustmentPanel 
                image={processedImage}
                onImageUpdate={updateProcessedImage}
              />
              
              <ColorAnalysis 
                analysis={colorAnalysis}
              />
              
              <EnhancePanel 
                image={processedImage}
                onImageUpdate={updateProcessedImage}
              />
              
              <PromptEditor 
                image={processedImage}
                onImageUpdate={updateProcessedImage}
              />
            </div>
          </div>
        )}
      </main>

      {/* Hidden canvas for download functionality */}
      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </div>
  );
};

export default App;