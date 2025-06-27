import streamlit as st
import torch
import rasterio
import numpy as np
import os
import zipfile
import tempfile
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import fitz  # PyMuPDF for PDF processing
from docx import Document  # python-docx for DOCX processing
import json
from forest_fire_model import UNet, RasterDataset

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
        # Process ZIP file containing rasters as before
        extract_to = os.path.join(upload_dir, "extracted")
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        # Initialize model and dataset
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet(in_channels=9, out_classes=4).to(device)
        model.eval()
        dataset = RasterDataset(extract_to)

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

    # Make prediction only for ZIP files
    if file_type == "ZIP (Rasters)":
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
                    os.path.join(extract_to, os.listdir(extract_to)[0]),
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
This application supports multiple file types for analysis:

1. ZIP files containing raster data for fire risk prediction
2. PDF files for text extraction and analysis
3. DOCX files for document analysis
4. CSV files for data analysis and visualization
5. JSON files for data viewing

The fire risk prediction model uses environmental factors as input:
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
