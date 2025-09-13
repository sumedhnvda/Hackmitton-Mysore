from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import os
import io
import base64
import json
from dotenv import load_dotenv
from pathlib import Path
import requests
from openai import OpenAI
import re
import cv2

# Ensure env is loaded from this backend directory regardless of CWD
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

app = FastAPI()

# CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DeepAI API key for pretrained colorization model
DEEPAI_API_KEY = os.getenv("DEEPAI_API_KEY", "quickstart-QUdJIGlzIGNvbWluZy4uLi4K")

# Configure OpenAI client if needed
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except:
    client = None  # Will use auto-colorize endpoint instead

@app.post("/enhance-image")
async def enhance_image(request: Request):
    """
    Scales an image by a given factor.
    """
    data = await request.json()
    image_data = data.get('image_data')
    scale_factor = data.get('scale_factor')
    
    if not image_data or not scale_factor:
        raise HTTPException(status_code=400, detail="Image and scale factor are required")
        
    try:
        # Convert base64 to PIL Image
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes))
        
        # Scale the image
        new_size = (img.width * scale_factor, img.height * scale_factor)
        resized_img = img.resize(new_size, Image.LANCZOS)
        
        # Convert back to base64
        buf = io.BytesIO()
        resized_img.save(buf, format='PNG')
        enhanced_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Analyze colors in the enhanced image
        img_array = np.array(resized_img)
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
            color_diversity = min(100, (avg_distance / 255) * 100)
            
            # Measure evenness of color distribution
            normalized_percentages = percentages / 100
            distribution_evenness = 100 * (1 - np.std(normalized_percentages))
            
            # Combine metrics
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
                "rgb": color.tolist(),
                "percentage": round(percentages[i], 1)
            })
        
        return {
            "success": True,
            "enhanced_image": enhanced_image,
            "dominant_colors": dominant_colors,
            "srgb_score": srgb_score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _compose_with_style(base_img: Image.Image, style_img: Image.Image):
    """Create a side-by-side composite (base | style) and a mask that edits only the left/base side.
    Returns: composite_img (RGB), mask_img (RGBA), left_ratio (float), base_size (w,h)
    """
    # Resize style to match base height
    h = base_img.height
    style_ratio = style_img.width / max(1, style_img.height)
    style_w = max(1, int(h * style_ratio))
    style_resized = style_img.resize((style_w, h), Image.LANCZOS)

    comp_w = base_img.width + style_w
    composite = Image.new('RGB', (comp_w, h))
    composite.paste(base_img, (0, 0))
    composite.paste(style_resized, (base_img.width, 0))

    # Mask: left (base) area transparent (to be edited), right area opaque (preserved)
    mask = Image.new('RGBA', (comp_w, h), (0, 0, 0, 255))
    left_transparent = Image.new('RGBA', (base_img.width, h), (0, 0, 0, 0))
    mask.paste(left_transparent, (0, 0))

    left_ratio = base_img.width / comp_w
    return composite, mask, left_ratio, (base_img.width, h)


@app.post("/edit-image")
async def edit_image(request: Request):
    """
    Edit image using OpenAI gpt-image-1. If a style_transfer_image is provided, perform
    style transfer by supplying a composite (base|style) image and a mask that preserves the
    right (style) side while regenerating the left (base) side.
    Returns the edited base image and color analysis.
    """
    if client is None:
        raise HTTPException(status_code=500, detail="OpenAI client not configured. Set OPENAI_API_KEY.")

    data = await request.json()
    image_data = data.get('image_data')
    prompt = (data.get('prompt') or '').strip()
    style_transfer_image = data.get('style_transfer_image')

    if not image_data or not prompt:
        raise HTTPException(status_code=400, detail="Image and prompt are required")

    try:
        # Decode base image
        base_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
        base_img = Image.open(io.BytesIO(base_bytes)).convert('RGB')

        edited_b64 = None

        if style_transfer_image:
            # Decode style image
            style_bytes = base64.b64decode(style_transfer_image.split(',')[1] if ',' in style_transfer_image else style_transfer_image)
            style_img = Image.open(io.BytesIO(style_bytes)).convert('RGB')

            composite, mask_img, left_ratio, base_size = _compose_with_style(base_img, style_img)

            comp_buf = io.BytesIO()
            composite.save(comp_buf, format='PNG')
            comp_buf.seek(0)

            mask_buf = io.BytesIO()
            mask_img.save(mask_buf, format='PNG')
            mask_buf.seek(0)

            # Strong, specific instruction for style transfer
            full_prompt = (
                "Transfer the artistic style, color palette, lighting, and texture from the right-side reference "
                "into the left image. Preserve the left image's content and structure (people, objects, layout). "
                "Do not modify the right side."
            )
            if prompt:
                full_prompt = prompt + " | " + full_prompt

            result = client.images.edits(
                model="gpt-image-1",
                image=comp_buf,
                mask=mask_buf,
                prompt=full_prompt,
                size="1024x1024",
                response_format="b64_json",
            )

            if not result or not getattr(result, 'data', None):
                raise HTTPException(status_code=500, detail="Failed to edit image with style transfer")

            out_b64 = result.data[0].b64_json
            # Crop left portion proportional to original base width
            out_bytes = base64.b64decode(out_b64)
            out_img = Image.open(io.BytesIO(out_bytes)).convert('RGB')
            crop_w = max(1, int(out_img.width * left_ratio))
            cropped = out_img.crop((0, 0, crop_w, out_img.height))
            # Resize back to original base size for seamless replacement
            cropped = cropped.resize(base_size, Image.LANCZOS)

            out_buf = io.BytesIO()
            cropped.save(out_buf, format='PNG')
            edited_b64 = base64.b64encode(out_buf.getvalue()).decode('utf-8')
        else:
            # Simple edit of the base image without style reference
            base_buf = io.BytesIO()
            base_img.save(base_buf, format='PNG')
            base_buf.seek(0)

            result = client.images.edits(
                model="gpt-image-1",
                image=base_buf,
                prompt=prompt,
                size="1024x1024",
                response_format="b64_json",
            )

            if not result or not getattr(result, 'data', None):
                raise HTTPException(status_code=500, detail="Failed to edit image")

            # Resize the edited image back to original base size
            out_b64 = result.data[0].b64_json
            out_bytes = base64.b64decode(out_b64)
            out_img = Image.open(io.BytesIO(out_bytes)).convert('RGB')
            resized = out_img.resize((base_img.width, base_img.height), Image.LANCZOS)
            out_buf = io.BytesIO()
            resized.save(out_buf, format='PNG')
            edited_b64 = base64.b64encode(out_buf.getvalue()).decode('utf-8')

        # Analyze colors on edited image
        try:
            edited_bytes = base64.b64decode(edited_b64)
            edited_img = Image.open(io.BytesIO(edited_bytes)).convert('RGB')
            img_array = np.array(edited_img)
            pixels = img_array.reshape(-1, 3)

            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_.astype(int)
            labels, counts = np.unique(kmeans.labels_, return_counts=True)
            percentages = (counts / pixels.shape[0]) * 100
            sorted_indices = np.argsort(percentages)[::-1]

            color_distances = []
            for i in range(len(colors)):
                for j in range(i + 1, len(colors)):
                    dist = np.sqrt(np.sum((colors[i] - colors[j]) ** 2))
                    color_distances.append(dist)
            if color_distances:
                avg_distance = np.mean(color_distances)
                color_diversity = min(100, (avg_distance / 255) * 100)
                normalized_percentages = percentages / 100
                distribution_evenness = 100 * (1 - np.std(normalized_percentages))
                srgb_score = round((color_diversity * 0.7 + distribution_evenness * 0.3), 1)
            else:
                srgb_score = 0.0

            dominant_colors = []
            for i in sorted_indices:
                c = colors[i]
                hex_code = f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"
                dominant_colors.append({
                    "hex": hex_code,
                    "rgb": c.tolist(),
                    "percentage": round(percentages[i], 1)
                })

            return {
                "success": True,
                "edited_image": edited_b64,
                "dominant_colors": dominant_colors,
                "srgb_score": srgb_score,
            }
        except Exception:
            # Return without analysis if it fails for any reason
            return {
                "success": True,
                "edited_image": edited_b64,
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-colors")
async def analyze_colors(request: Request):
    """
    Analyzes the colors in an image to find the top 5 dominant colors and calculate sRGB coverage.
    """
    data = await request.json()
    image_data = data.get('image_data')
    
    if not image_data:
        raise HTTPException(status_code=400, detail="Image data is required")
        
    try:
        # Convert base64 to PIL Image
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_array = np.array(img)
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
            "success": True,
            "dominant_colors": dominant_colors,
            "srgb_score": srgb_score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/colorize-image")
async def colorize_image(request: Request):
    """
    Colorizes an image using the auto-colorization model.
    """
    try:
        data = await request.json()
        image_data = data.get('image_data')
        
        if not image_data:
            raise HTTPException(status_code=400, detail="Image data is required")
            
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        # Convert base64 to PIL Image
        image_bytes = base64.b64decode(image_data)
        original_image = Image.open(io.BytesIO(image_bytes))
        
        # Forward the request to the auto-colorization service
        colorize_response = requests.post(
            'http://127.0.0.1:8002/api/auto-colorize',
            json={'image_data': image_data}
        )
        
        if colorize_response.status_code == 200:
            return colorize_response.json()
        else:
            raise HTTPException(
                status_code=colorize_response.status_code,
                detail="Failed to colorize image"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pretrained-colorize")
async def pretrained_colorize(request: Request):
    """
    Colorizes a grayscale image using OpenCV DNN colorization model (no external API).
    Expects OpenCV model files in backend/model_files:
    - colorization_deploy_v2.prototxt
    - colorization_release_v2.caffemodel
    - pts_in_hull.npy
    """
    try:
        data = await request.json()
        image_data = data.get('image_data')

        if not image_data:
            raise HTTPException(status_code=400, detail="Image data is required")

        # Base64 decode input image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_rgb = np.array(pil_img)

        # Load OpenCV colorization model
        model_dir = Path(__file__).parent / 'model_files'
        proto = str(model_dir / 'colorization_deploy_v2.prototxt')
        model = str(model_dir / 'colorization_release_v2.caffemodel')
        pts = str(model_dir / 'pts_in_hull.npy')
        if not (Path(proto).exists() and Path(model).exists() and Path(pts).exists()):
            raise HTTPException(status_code=500, detail="OpenCV colorization model files not found in backend/model_files")

        net = cv2.dnn.readNetFromCaffe(proto, model)
        pts_in_hull = np.load(pts)
        pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)
        net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]
        net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, dtype=np.float32)]

        # Convert to Lab and prepare L channel
        img_rgb_float = img_rgb.astype(np.float32) / 255.0
        img_lab = cv2.cvtColor(img_rgb_float, cv2.COLOR_RGB2LAB)
        img_l = img_lab[:, :, 0]
        img_rs = cv2.resize(img_rgb_float, (224, 224))
        img_lab_rs = cv2.cvtColor(img_rs, cv2.COLOR_RGB2LAB)
        l_rs = img_lab_rs[:, :, 0]
        l_rs -= 50  # mean-centering as expected by the model

        net.setInput(cv2.dnn.blobFromImage(l_rs))
        ab_dec = net.forward()[0, :, :, :].transpose((1, 2, 0))  # HxWx2
        ab_dec_us = cv2.resize(ab_dec, (img_rgb.shape[1], img_rgb.shape[0]))

        lab_out = np.concatenate((img_l[:, :, np.newaxis], ab_dec_us), axis=2)
        bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_Lab2BGR)
        bgr_out = np.clip(bgr_out, 0, 1)
        rgb_out = cv2.cvtColor((bgr_out * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        colorized_image = Image.fromarray(rgb_out)

        # Analyze colors in the colorized image
        img_array = np.array(colorized_image)
        pixels = img_array.reshape(-1, 3)
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        percentages = (counts / pixels.shape[0]) * 100
        sorted_indices = np.argsort(percentages)[::-1]

        color_distances = []
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                dist = np.sqrt(np.sum((colors[i] - colors[j]) ** 2))
                color_distances.append(dist)
        if color_distances:
            avg_distance = np.mean(color_distances)
            color_diversity = min(100, (avg_distance / 255) * 100)
            normalized_percentages = percentages / 100
            distribution_evenness = 100 * (1 - np.std(normalized_percentages))
            srgb_score = round((color_diversity * 0.7 + distribution_evenness * 0.3), 1)
        else:
            srgb_score = 0.0

        dominant_colors = []
        for i in sorted_indices:
            color = colors[i]
            hex_code = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            dominant_colors.append({
                "hex": hex_code,
                "rgb": color.tolist(),
                "percentage": round(percentages[i], 1),
            })

        buffer = io.BytesIO()
        colorized_image.save(buffer, format='PNG')
        colorized_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return {
            "success": True,
            "colorized_image": colorized_base64,
            "dominant_colors": dominant_colors,
            "srgb_score": srgb_score,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
