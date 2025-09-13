from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import base64
import torch
import torch.nn as nn
from pathlib import Path
from skimage.color import rgb2lab, lab2rgb
from sklearn.cluster import KMeans

app = FastAPI()

# CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model architecture
class EfficientColorizationNet(nn.Module):
    def __init__(self):
        super(EfficientColorizationNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 3, padding=1), nn.Tanh()
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


MODEL_PATH = Path(__file__).parent / "efficient_colorization_5k.pth"
#MODEL_PATH = Path(__file__).parent / "backend/efficient_colorization_5k_negligible.pth"

try:
    # Create model instance
    model = EfficientColorizationNet()
    
    # Load checkpoint with map_location to handle CPU/GPU differences
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    model = model.to(device)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def preprocess_image(image):
    """Convert image to LAB color space and prepare L channel for model"""
    # Resize image to expected size
    image = image.resize((128, 128), Image.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0
    
    # Convert to LAB color space and get L channel
    L = rgb2lab(img_array)[:, :, 0] / 100.0
    
    # Convert to tensor
    L_tensor = torch.from_numpy(L).unsqueeze(0).unsqueeze(0).float().to(device)
    
    print(f"Input tensor shape: {L_tensor.shape}")  # Debugging
    print(f"Input tensor range: [{L_tensor.min():.3f}, {L_tensor.max():.3f}]")  # Debugging
    
    return L_tensor

def postprocess_image(L_tensor, AB_tensor, original_size):
    """Convert L and AB channels back to RGB image"""
    # Convert tensors to numpy arrays
    L_np = L_tensor.squeeze().cpu().numpy() * 100.0
    AB_np = AB_tensor.squeeze().cpu().numpy().transpose(1, 2, 0) * 128.0
    
    # Combine L and AB channels
    lab_img = np.dstack([L_np, AB_np])
    
    # Convert from LAB to RGB
    rgb_img = lab2rgb(lab_img)
    
    # Scale to 0-255 range and convert to uint8
    rgb_img = np.clip(rgb_img, 0, 1) * 255
    rgb_img = rgb_img.astype(np.uint8)
    
    # Convert to PIL Image
    output_img = Image.fromarray(rgb_img)
    
    # Resize back to original size if needed
    if original_size != (128, 128):
        output_img = output_img.resize(original_size, Image.LANCZOS)
    
    return output_img

def analyze_colors(image):
    """
    Analyzes colors in an image using K-means clustering
    Returns dominant colors and sRGB coverage score
    """
    # Convert image to numpy array
    img_array = np.array(image)
    pixels = img_array.reshape(-1, 3)
    
    # Use k-means to find the 5 most dominant colors
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Get the colors and their percentages
    colors = kmeans.cluster_centers_.astype(int)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    percentages = (counts / pixels.shape[0]) * 100
    
    # Sort colors by percentage
    sorted_indices = np.argsort(percentages)[::-1]
    
    # Calculate sRGB coverage score
    # This is a simplified measure of color richness based on color diversity
    # We measure distance between clusters and their distribution
    color_distances = []
    for i in range(len(colors)):
        for j in range(i+1, len(colors)):
            # Euclidean distance between colors in RGB space
            dist = np.sqrt(np.sum((colors[i] - colors[j])**2))
            color_distances.append(dist)
    
    # Calculate a score based on average distance and weighted by color distribution
    if color_distances:
        avg_distance = np.mean(color_distances)
        # Normalize to 0-100 range with a reasonable scaling factor
        # Higher distance means better color separation
        color_diversity = min(100, (avg_distance / 255) * 100)
        
        # Measure evenness of color distribution (entropy-like)
        normalized_percentages = percentages / 100
        distribution_evenness = 100 * (1 - np.std(normalized_percentages))
        
        # Combine metrics (equal weights)
        srgb_score = round((color_diversity * 0.7 + distribution_evenness * 0.3), 1)
    else:
        srgb_score = 0.0
    
    # Format results for the frontend
    dominant_colors = []
    for i in sorted_indices:
        color = colors[i]
        hex_code = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        dominant_colors.append({
            "hex": hex_code,
            "rgb": color.tolist(),  # Include RGB values as list
            "percentage": round(percentages[i], 1)
        })
        
    return {
        "dominant_colors": dominant_colors,
        "srgb_score": srgb_score
    }

@app.post("/colorize-image")
async def colorize_image(request: Request):
    """
    Automatically colorizes an uploaded image using the efficient_colorization_5k model.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    try:
        data = await request.json()
        image_data = data.get('image_data')
        
        if not image_data:
            raise HTTPException(status_code=400, detail="Image data is required")
            
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        # Convert base64 to PIL Image
        try:
            image_bytes = base64.b64decode(image_data)
            original_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
        
        # Keep track of original size for resizing back
        original_size = original_image.size
        
        try:
            # Preprocess image to get L channel
            L_tensor = preprocess_image(original_image)
            
            # Run inference to get AB channels
            with torch.no_grad():
                AB_tensor = model(L_tensor)
            
            # Postprocess output to get colorized image
            colorized_image = postprocess_image(L_tensor, AB_tensor, original_size)
            
            # Analyze colors in the colorized image
            color_analysis = analyze_colors(colorized_image)
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                raise HTTPException(status_code=500, detail="GPU out of memory. Try with a smaller image.")
            raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")
        
        # Convert back to base64
        buffer = io.BytesIO()
        colorized_image.save(buffer, format='PNG')
        colorized_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            "success": True,
            "colorized_image": colorized_base64,
            "dominant_colors": color_analysis["dominant_colors"],
            "srgb_score": color_analysis["srgb_score"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)  # Using port 8002
