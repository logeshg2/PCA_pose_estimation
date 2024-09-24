# imports
import cv2
import torch
import numpy as np
import open3d as o3d
from ultralytics import YOLO
import matplotlib.pyplot as plt
from depth_anything_v2.dpt import DepthAnythingV2

# selecting device for torch (NN) -> here Nvidia Cuda is used
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# depth_anything_v2_small -> vits
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}
encoder = 'vits'

# load the pretrained depth model
depth_model = DepthAnythingV2(**model_configs[encoder])
depth_model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
depth_model = depth_model.to(DEVICE).eval()

# load the custom trained yolo model for {bottle and can} dataset
yolo_model = YOLO("./checkpoints/best.pt")

# for now a single image is read and performed detection
rgb_img = cv2.imread(".//data//image.png")

# predict the input rgb image:
result = (yolo_model.predict(source=rgb_img,save=False,imgsz=640,conf=0.5,show=False))[0].boxes
xyxy = list(map(int,result.xyxy[0].tolist()))
image = cv2.rectangle(rgb_img, (xyxy[0],xyxy[1]), (xyxy[2],xyxy[3]), (0,0,255), 2)
plt.imshow(image)
plt.show()

# estimate the depth
depth = depth_model.infer_image(rgb_img)
plt.imshow(depth)
plt.show()


# Point Cloud from rgb and depth image:
height,width = rgb_img.shape[0:2]
depth_min, depth_max = np.min(depth), np.max(depth)
resized_pred = depth_normalized = (depth_max - depth) / (depth_max - depth_min)
# focal lengths         # usually calculated from intrinsic_parameters
focal_length_x = 535.4
focal_length_y = 539.2
# Generate mesh grid and calculate point cloud coordinates
x, y = np.meshgrid(np.arange(width), np.arange(height))
x = (x - width / 2) / focal_length_x
y = (y - height / 2) / focal_length_y
z = np.array(resized_pred)
points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
colors = np.array(rgb_img).reshape(-1, 3) / 255.0
# Create the point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])


# Crop the point cloud -> detected object:
croped_points = points.reshape([height,width,3])[xyxy[1]:xyxy[3]+1, xyxy[0]:xyxy[2]+1,:]
croped_points = croped_points.reshape(croped_points.shape[0] * croped_points.shape[1],3)
croped_colors = colors.reshape([height,width,3])[xyxy[1]:xyxy[3]+1, xyxy[0]:xyxy[2]+1,:]
croped_colors = croped_colors.reshape(croped_colors.shape[0] * croped_colors.shape[1],3)
# print(xyxy)
# print(points.reshape([height,width,3]).shape)
croped_pcd = o3d.geometry.PointCloud()
croped_pcd.points = o3d.utility.Vector3dVector(croped_points)
croped_pcd.colors = o3d.utility.Vector3dVector(croped_colors)
o3d.visualization.draw_geometries([croped_pcd])


## TODO ##
# To implement PCA pose estimation
# To calculate camera to image point translation
# To integrate both the rotation and translation (6D Pose)
# To pass the base to object point to the robot (after base to camera transformation) 