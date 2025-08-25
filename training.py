import torch
import torch.nn as nn
import numpy as np
import nibabel as nib

# ----------------------------
# Hyperparameters
# ----------------------------
EPOCHS = 8
LEARNING_RATE = 1e-4

# ----------------------------
# Model definition
# ----------------------------
class BasicConvolution(nn.Module):
    def __init__(self, input_ch: int, output_ch: int) -> None:
        super().__init__()

        self.activation    = nn.ReLU()
        self.convolution   = nn.Conv3d(input_ch, output_ch, kernel_size=5, padding='same')
        self.normalization = nn.BatchNorm3d(output_ch)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolution(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x

class Basic3DModel(nn.Module):
    def __init__(self, input_ch: int, output_ch: int) -> None:
        super().__init__()

        self.layer_1 = BasicConvolution(input_ch, 6)
        self.layer_2 = BasicConvolution(6, 12)
        self.layer_3 = nn.Conv3d(12, output_ch, kernel_size=1, padding='same')
        self.out_act = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.out_act(x)
        return x


# ----------------------------
# Main script
# ----------------------------
if __name__ == "__main__":
    print("hello World")

    # ----------------------------
    # Load dataset (replace with your own paths)
    # ----------------------------
    # NOTE: Do not upload dataset files to GitHub. 
    # Provide instructions in README instead.
    input_x = nib.load('./path/to/t1_image.nii.gz').get_fdata()
    input_y = nib.load('./path/to/segmentation_mask.nii.gz').get_fdata()

    # Debug: check shapes and unique values of input MRI data
    print(input_x.shape)
    print(np.unique(input_x))
    
    # Debug: check shapes and unique values of segmentation mask
    print(input_y.shape)
    print(np.unique(input_y))

    # ----------------------------
    # Preprocessing MRI image
    # ----------------------------
    input_x = (input_x - np.min(input_x)) / np.ptp(input_x)  # normalize
    input_x = torch.from_numpy(input_x)
    input_x = torch.unsqueeze(input_x, dim=0)  # add channel dimension
    input_x = torch.unsqueeze(input_x, dim=0)  # add batch dimension
    input_x = input_x.float()

    # ----------------------------
    # Preprocessing segmentation mask
    # ----------------------------
    input_y = np.where(input_y == 4, 3, input_y)  # re-map class 4 → class 3
    data_cls = [
        np.where(input_y == 0, 1, 0),  # class 0 mask
        np.where(input_y == 1, 1, 0),  # class 1 mask
        np.where(input_y == 2, 1, 0),  # class 2 mask
        np.where(input_y == 3, 1, 0),  # class 3 mask
    ]
    input_y = np.stack(data_cls)
    input_y = torch.from_numpy(input_y)
    input_y = torch.unsqueeze(input_y, dim=0)  # add batch dimension
    input_y = input_y.float()
    
    # Debug: check shapes and unique values after preprocessing
    print(input_x.shape)
    print(torch.unique(input_x))
    print(input_y.shape)
    print(torch.unique(input_y))
    
    # ----------------------------
    # Model and training setup
    # ----------------------------
    x = torch.rand(1, 1, 240, 240, 155)  # dummy input for debugging
    model = Basic3DModel(1, 4)
    y = model(x)
    
    # Debug: check model output shape and value range
    print(y.shape)
    print(torch.unique(y))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    lossFunction = torch.nn.MSELoss()
    
    # ----------------------------
    # Training loop
    # ----------------------------
    for epoch_count in range(EPOCHS):
        print("Epoch :", epoch_count)
        
        model.train()  # set the model to train mode      
        output = model(input_x)
        loss = lossFunction(output, input_y)
        loss.backward()
        optimizer.step()

        # Debug: print loss every epoch
        print(loss)
        print(" ")

    print("COMPLETE")
