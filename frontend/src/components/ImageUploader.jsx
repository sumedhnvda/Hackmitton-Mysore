import React, { useState, useRef } from 'react';

const ImageUploader = ({ onImageUpload, uploadedImage }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  const handleFileUpload = async (file) => {
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file');
      return;
    }

    setIsUploading(true);
    
    try {
      // Convert file to base64 for preview
      const reader = new FileReader();
      reader.onload = (e) => {
        const base64 = e.target.result.split(',')[1]; // Remove data:image/...;base64, prefix
        onImageUpload({
          file,
          base64,
          preview: e.target.result,
          filename: file.name
        });
      };
      reader.readAsDataURL(file);
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Error uploading file. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="image-uploader">
      <div 
        className={`upload-area ${isDragging ? 'dragging' : ''} ${uploadedImage ? 'has-image' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
        style={{
          border: '2px dashed #ccc',
          borderRadius: '10px',
          padding: '40px',
          textAlign: 'center',
          cursor: 'pointer',
          backgroundColor: isDragging ? '#f0f0f0' : '#fafafa',
          minHeight: '200px',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center'
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />
        
        {uploadedImage ? (
          <div className="image-preview" style={{ textAlign: 'center' }}>
            <img 
              src={uploadedImage.preview} 
              alt="Uploaded" 
              style={{ 
                maxWidth: '300px', 
                maxHeight: '200px', 
                borderRadius: '8px',
                marginBottom: '10px'
              }}
            />
            <div className="image-info">
              <p style={{ margin: '5px 0', fontWeight: 'bold' }}>{uploadedImage.filename}</p>
              <button 
                onClick={(e) => {
                  e.stopPropagation();
                  onImageUpload(null);
                }}
                style={{
                  padding: '5px 15px',
                  backgroundColor: '#ff4444',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                Remove
              </button>
            </div>
          </div>
        ) : (
          <div className="upload-prompt">
            {isUploading ? (
              <div className="uploading">
                <div className="spinner" style={{
                  width: '40px',
                  height: '40px',
                  border: '4px solid #f3f3f3',
                  borderTop: '4px solid #3498db',
                  borderRadius: '50%',
                  animation: 'spin 1s linear infinite',
                  margin: '0 auto 20px'
                }}></div>
                <p>Uploading...</p>
              </div>
            ) : (
              <>
                <div className="upload-icon" style={{ fontSize: '48px', marginBottom: '20px' }}>📁</div>
                <h3 style={{ margin: '10px 0' }}>Drag & Drop Image</h3>
                <p style={{ margin: '5px 0', color: '#666' }}>or click to browse</p>
                <small style={{ color: '#999' }}>Supports: JPG, PNG, GIF, WebP</small>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageUploader;