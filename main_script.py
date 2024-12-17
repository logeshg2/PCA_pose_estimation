# imports
import cv2
import math
import torch
import numpy as np
import open3d as o3d
import transformations as T
from ultralytics import YOLO
import matplotlib.pyplot as plt
from depth_anything_v2.dpt import DepthAnythingV2

### Initialize parameters ###
# selecting device for torch (NN) -> here Nvidia Cuda is used
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# depth_anything_v2_small -> vits
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]}
}
encoder = 'vitb'

# variables from camera calibrations
fx,fy = 607.056, 606.697
cx,cy = 328.384, 241.043


### Loding Models ###
# load the pretrained depth model
depth_model = DepthAnythingV2(**model_configs[encoder])
depth_model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
depth_model = depth_model.to(DEVICE).eval()

# load the custom trained yolo model for {bottle and can} dataset
yolo_model = YOLO("./checkpoints/best_shape.pt")


### Read example images ###
# for now a single image is read and performed detection
rgb_img = cv2.imread(".//data//me.png")
# convert into gray scale
gray = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2GRAY)
gray = np.stack((gray,gray,gray),axis=-1)
rgb_img = gray


### Object Detection ###
# predict the input rgb image:
result = (yolo_model.predict(source=rgb_img,save=False,imgsz=640,conf=0.5,show=False))[0].boxes
# print(result)
xyxy = list(map(int,result.xyxy[5].tolist()))
xywh = list(map(int,result.xywh[5].tolist()))
image = cv2.rectangle(rgb_img, (xyxy[0],xyxy[1]), (xyxy[2],xyxy[3]), (0,0,255), 2)
plt.imshow(image)
plt.show()


### Depth Estimaiton ###
# estimate the depth
depth_img = depth_model.infer_image(np.array(rgb_img))
# print(depth_img)
depth = np.asanyarray(depth_img)
plt.imshow(depth)
plt.show()


### Point Cloud Generation ###
height,width = rgb_img.shape[0:2]
pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
rgb = o3d.geometry.Image(rgb_img)
depth = o3d.geometry.Image(depth)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, convert_rgb_to_intensity=False)
temp = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, pinhole_camera_intrinsic) 
pcd = o3d.geometry.PointCloud()
pcd.points=temp.points
pcd.colors=temp.colors
points = np.array(pcd.points)
colors = np.array(pcd.colors)
# print(points.shape)
o3d.visualization.draw_geometries([pcd])


### Crop the point cloud -> detected object: ###
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


### PCA using SVD ###
print('\nRotation along z-axis')
# svd based pose
X = np.asarray(croped_pcd.points).T
# print(X.shape)
nPoints = max(X.shape)
Xavg = np.mean(X,axis=1)
B = X - np.tile(Xavg,(nPoints,1)).T
U, S, VT = np.linalg.svd(B/np.sqrt(nPoints),full_matrices=0)
print(U,'\n') # z-axis rotation matrix


### euler angle calculation ###
# def rotationMatrixToEulerAngles(R) :
#     sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
#     singular = sy < 1e-6
#     if  not singular :
#         x = math.atan2(R[2,1] , R[2,2])
#         y = math.atan2(-R[2,0], sy)
#         z = math.atan2(R[1,0], R[0,0])
#     else :
#         x = math.atan2(-R[1,2], R[1,1])
#         y = math.atan2(-R[2,0], sy)
#         z = 0
#     return np.array([x, y, z])
# eul = np.rad2deg(rotationMatrixToEulerAngles(U))
# print(eul,'\n')
# print("Orientation in z axis: ",eul[2])


### position calculation ###
def find_position(u,v,z):
    z = z * 1000            # meter to mm
    X = ((u - cx) * z) / fx
    Y = ((v - cy) * z) / fy
    Z = z
    return [X,Y,Z]


### Homogeneous transformation matrix ###
HTM = np.eye(4,4)
HTM[0:3,0:3] = U
XYZ = find_position(xywh[0],xywh[1],depth_img[xywh[0]][xywh[1]])
HTM[0:3,3] =  XYZ
print('\nHomogeneous Transformation Matrix:')
print('\n',HTM,'\n')


### drawing frame axes ###
cam_params = np.array([[fx,0,cx],[0,fy,cy],[0,0,1.0]])
cam_dist = np.array([])
tvec = np.array(XYZ)
cv2.drawFrameAxes(image,cam_params,cam_dist,U,tvec,300,5)
plt.imshow(image)
plt.show()