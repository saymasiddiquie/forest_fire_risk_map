import streamlit as st
import os
import zipfile
import tempfile
import pandas as pd
import fitz  # PyMuPDF for PDF processing
from docx import Document  # python-docx for DOCX processing
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import rasterio

# Color map for visualization
def create_fire_risk_color_map():
    cmap = plt.get_cmap('viridis')
    def color_func(value):
        norm_val = (value - 0) / (255 - 0)
        return (cmap(norm_val) * 255).astype(np.uint8)
    return color_func

color_map = create_fire_risk_color_map()

# Initialize Streamlit app
st.title("Forest Fire Risk Analysis")

# Create a temporary directory for file uploads
upload_dir = "uploads"
os.makedirs(upload_dir, exist_ok=True)

# File upload section
st.header("Upload Files")
st.write("Upload different types of files for analysis:")

# File upload options
file_type = st.selectbox(
    "Select file type",
    ["ZIP (Rasters)", "PDF", "DOCX", "CSV", "JSON"]
)

uploaded_file = st.file_uploader(
    "Choose a file",
    type=["zip", "pdf", "docx", "csv", "json"]
)

if uploaded_file is not None:
    # Save uploaded file
    temp_file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Process file based on type
    if file_type == "ZIP (Rasters)":
        # Process ZIP file containing rasters
        extract_to = os.path.join(upload_dir, "extracted")
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        # Get first raster file for visualization
        raster_files = [f for f in os.listdir(extract_to) if f.endswith('.tif')]
        if raster_files:
            sample_raster = raster_files[0]
            with rasterio.open(os.path.join(extract_to, sample_raster)) as src:
                # Read first band as sample
                sample_data = src.read(1)
                
                # Create a simple visualization
                norm_data = (sample_data - sample_data.min()) / (sample_data.max() - sample_data.min())
                rgb_data = color_map(sample_data)
                
                st.header("Sample Raster Visualization")
                st.image(rgb_data, caption=f"Visualization of {sample_raster}", use_column_width=True)

    elif file_type == "PDF":
        # Process PDF file
        doc = fitz.open(temp_file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        
        st.header("PDF Content Preview")
        st.text_area("Extracted Text", text, height=200)

    elif file_type == "DOCX":
        # Process DOCX file
        doc = Document(temp_file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        
        st.header("DOCX Content Preview")
        st.text_area("Extracted Text", text, height=200)

    elif file_type == "CSV":
        # Process CSV file
        df = pd.read_csv(temp_file_path)
        st.header("CSV Data Preview")
        st.dataframe(df.head())
        
        # Add basic analysis
        st.header("Data Statistics")
        st.write(df.describe())

    elif file_type == "JSON":
        # Process JSON file
        with open(temp_file_path, 'r') as f:
            data = json.load(f)
        
        st.header("JSON Data Preview")
        st.json(data)

# Add about section
st.sidebar.header("About")
st.sidebar.info("""
This application supports multiple file types for analysis:

1. ZIP files containing raster data
   - Visualizes sample raster data
   - Shows basic statistics

2. PDF files
   - Extracts and displays text content
   - Preserves document formatting

3. DOCX files
   - Extracts and displays document text
   - Preserves paragraph structure

4. CSV files
   - Displays data in tabular format
   - Shows basic statistics
   - Data visualization

5. JSON files
   - Displays structured data
   - Easy to read format

The application provides a simple way to analyze and visualize different types of files without requiring complex processing.
""")

if uploaded_file is not None:
    # Save uploaded file to temp directory
    temp_zip_path = os.path.join(upload_dir, "uploaded.zip")
    with open(temp_zip_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Extract the ZIP file
    extract_to = os.path.join(upload_dir, "extracted")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=9, out_classes=4).to(device)
    model.eval()  # Set model to evaluation mode

    # Create dataset
    dataset = RasterDataset(extract_to)
    
    # Make prediction
    if st.button("Generate Prediction"):
        try:
            # Get input data
            input_data = dataset[0].to(device)
            
            # Make prediction
            with torch.no_grad():
                prediction = model(input_data)
                prediction = prediction.squeeze().cpu().numpy()
                
            # Create output directory
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save prediction as GeoTIFF
            with rasterio.open(
                os.path.join(extract_to, os.listdir(extract_to)[0]),  # Use any existing file for metadata
                "r"
            ) as src:
                meta = src.meta
                meta.update(
                    dtype=rasterio.uint8,
                    count=1
                )
                
                prediction_path = os.path.join(output_dir, "fire_risk_prediction.tif")
                with rasterio.open(prediction_path, 'w', **meta) as dst:
                    dst.write(prediction.astype(rasterio.uint8), 1)

            # Display prediction
            st.header("Fire Risk Prediction")
            
            # Create a color map for visualization
            cmap = plt.get_cmap('viridis')
            
            # Normalize prediction for visualization
            norm_pred = (prediction - prediction.min()) / (prediction.max() - prediction.min())
            
            # Convert to RGB for display
            rgb_pred = (cmap(norm_pred) * 255).astype(np.uint8)
            
            # Display the prediction
            st.image(rgb_pred, caption="Fire Risk Prediction Map", use_column_width=True)
            
            # Add download button
            with open(prediction_path, "rb") as f:
                st.download_button(
                    label="Download Prediction",
                    data=f,
                    file_name="fire_risk_prediction.tif",
                    mime="image/tiff"
                )
            
            st.success("Prediction generated successfully!")
            
        except Exception as e:
            st.error(f"Error generating prediction: {str(e)}")

# Add about section
st.sidebar.header("About")
st.sidebar.info("""
This application predicts forest fire risk using a U-Net model.
The model takes multiple environmental factors as input:
- Aspect
- Humidity
- Land Use/Land Cover (LULC)
- Rainfall
- Settlement Distance
- Slope
- Temperature
- Wind Direction
- Wind Speed
""")
