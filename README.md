# VINCI - AI Image Colorization Platform

This repository contains VINCI, an AI-powered image colorization platform developed for Hackmitton 2.0 Mysore Event.

## Project Structure

```
├── backend/                    # FastAPI backend services
│   ├── auto_colorize.py       # Main colorization service (port 8002)
│   ├── main.py               # Color analysis service (port 8000)
│   ├── efficient_colorization_5k.pth  # PyTorch model
│   ├── experiments/          # Development and testing files
│   │   ├── sample_images/    # Test images
│   │   ├── test.py          # Testing scripts
│   │   └── Test.ipynb       # Jupyter notebook experiments
│   └── requirements.txt     # Python dependencies
├── frontend/                 # React frontend application
│   ├── src/                 # Source code
│   │   ├── components/      # React components
│   │   └── styles/         # CSS styles
│   ├── package.json        # Node.js dependencies
│   └── vite.config.js      # Vite configuration
└── .gitignore              # Git ignore patterns
```

## Features

- **AI-Powered Colorization**: Custom PyTorch model for black & white image colorization
- **Color Analysis**: Advanced color palette analysis and sRGB scoring
- **Image Enhancement**: Built-in image adjustment tools
- **Modern UI**: React-based responsive interface
- **Real-time Processing**: Fast image processing with GPU support

## Setup Instructions

### Backend Setup
1. Navigate to backend directory: `cd backend`
2. Install dependencies: `pip install -r requirements.txt`
3. Start colorization service: `python auto_colorize.py` (runs on port 8002)
4. Start analysis service: `python main.py` (runs on port 8000)

### Frontend Setup
1. Navigate to frontend directory: `cd frontend`
2. Install dependencies: `npm install`
3. Start development server: `npm run dev`

## Usage

1. Upload a black & white image
2. Click "Custom Colorize" to process with the AI model
3. Use adjustment panels to fine-tune results
4. Download the colorized image

## Technology Stack

- **Backend**: FastAPI, PyTorch,scikit-learn
- **Frontend**: React, Vite, modern CSS
- **AI Model**: Custom efficient colorization neural network

## Development

This project was developed for the Hackmitton 2.0 Mysore hackathon event, focusing on AI-powered image processing and modern web technologies. 