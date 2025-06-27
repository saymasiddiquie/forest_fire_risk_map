
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
import os
from torchvision import transforms
import zipfile

# 1. Extract Input ZIP

zip_path = "C:/Users/Sayama Siddiquie/Downloads/Forest_fire_risk_map/stacked_inputs_sample.zip"

extract_to = "./extracted_inputs"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("Extracted files:", os.listdir(os.path.join(extract_to, "stacked_inputs")))


# 2. U-NET ARCHITECTURE

class UNet(nn.Module):
    def __init__(self, in_channels=9, out_classes=4):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_classes = out_classes
        
        # Print input channels for debugging
        print(f"Model initialized with {in_channels} input channels")

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x):
        # Ensure input has shape (batch, channels, height, width)
        if len(x.shape) == 3:  # Add batch dimension if missing
            x = x.unsqueeze(0)
        elif len(x.shape) == 4 and x.shape[1] != self.in_channels:  # If channels are in last dimension
            x = x.permute(0, 3, 1, 2)
        
        # Print shape for debugging
        print(f"Input shape: {x.shape}")
        
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.upconv4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))
        return self.final(d1)

# 3. CUSTOM DATASET

class RasterDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        self.image_stack = []
        for file in sorted(os.listdir(image_dir)):
            if file.endswith(".tif"):
                with rasterio.open(os.path.join(image_dir, file)) as src:
                    data = src.read(1).astype(np.float32)
                    # Normalize each band to 0-1 range
                    min_val = np.min(data)
                    max_val = np.max(data)
                    if max_val > min_val:
                        data = (data - min_val) / (max_val - min_val)
                    self.image_stack.append(data)
        
        # Stack channels and ensure shape is (channels, height, width)
        self.image_stack = np.stack(self.image_stack)
        
        # Get reference file for metadata
        self.ref_file = os.path.join(image_dir, os.listdir(image_dir)[0])
        
        # Ensure number of channels matches model expectations
        num_channels = self.image_stack.shape[0]
        print(f"Found {num_channels} input channels in the dataset")
        if num_channels != 9:
            raise ValueError(f"Expected 9 input channels but got {num_channels}")
            
    def __getitem__(self, idx):
        image = np.array(self.image_stack, dtype=np.float32)
        # Ensure shape is (channels, height, width)
        if len(image.shape) == 3:  # If shape is (height, width, channels)
            image = np.transpose(image, (2, 0, 1))
        
        # Convert to torch tensor and add batch dimension
        image = torch.tensor(image).unsqueeze(0)
        
        if self.transform:
            image = self.transform(image)
        
        # Ensure channels are in the second dimension
        if image.shape[2] == self.image_stack.shape[0]:
            image = image.permute(0, 3, 1, 2)
        
        return image

    def __len__(self):
        return 1
        return image.unsqueeze(0)  # Add batch dimension



# 4. PREDICTION FUNCTION
def predict(model, input_data, device):
    model.eval()
    with torch.no_grad():
        # Move input to device
        inputs = input_data.to(device)
        outputs = model(inputs)
        # Get the predicted class
        _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()

# 5. PREDICTION FUNCTION
def predict_and_save(model, input_image, out_path, ref_file):
    model.eval()
    with torch.no_grad():
        logits = model(input_image)  # No extra dimension!
        prediction = torch.argmax(logits.squeeze(0), dim=0).cpu().numpy()

        with rasterio.open(ref_file) as src:
            meta = src.meta.copy()
        meta.update({"count": 1, "dtype": "uint8"})

        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(prediction.astype(rasterio.uint8), 1)

#  USAGE / MAIN

if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs(extract_to, exist_ok=True)
    os.makedirs("./output", exist_ok=True)
    
    # Extract ZIP file
    print("Extracting ZIP file...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    # Setup paths
    image_dir = os.path.join(extract_to, "stacked_inputs")  # Folder with .tif raster inputs
    
    # Initialize dataset
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float() / 255.0)  # Normalize to [0,1] range
    ])
    
    dataset = RasterDataset(image_dir, transform=transform)
    
    # Get the input data directly
    input_data = dataset[0]  # Already has batch dimension from dataset
    
    # Initialize model and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    
    # Make prediction
    print("Making prediction...")
    prediction = predict(model, input_data, device)
    
    # Save the prediction as a GeoTIFF
    # Use one of the existing raster files for metadata
    raster_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
    if not raster_files:
        raise FileNotFoundError(f"No .tif files found in {image_dir}")
    
    with rasterio.open(os.path.join(image_dir, raster_files[0])) as src:
        meta = src.meta
        meta.update(
            dtype=rasterio.uint8,
            count=1  # Single channel output
        )
        
        # Ensure prediction is in the correct shape (height, width)
        prediction = prediction.squeeze()  # Remove batch and channel dimensions
        
        with rasterio.open("./output/prediction.tif", "w", **meta) as dst:
            dst.write(prediction.astype(rasterio.uint8), 1)
    
    print(f"Prediction saved to: ./output/prediction.tif")
    predict_and_save(model, input_data, "fire_prediction_map.tif", os.path.join(image_dir, os.listdir(image_dir)[0]))
