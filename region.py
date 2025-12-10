import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import mediapipe as mp

class RegionFeatureExtractor:    
    def __init__(self, static_mode=False):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_mode,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.REGION_LANDMARKS = {
        'LEFT_EYE': list(set(itertools.chain(*self.mp_face_mesh.FACEMESH_LEFT_EYE))),
        'RIGHT_EYE': list(set(itertools.chain(*self.mp_face_mesh.FACEMESH_RIGHT_EYE))),
        'LEFT_EYEBROW': list(set(itertools.chain(*self.mp_face_mesh.FACEMESH_LEFT_EYEBROW))),
        'RIGHT_EYEBROW': list(set(itertools.chain(*self.mp_face_mesh.FACEMESH_RIGHT_EYEBROW))),
        'LIPS': list(set(itertools.chain(*self.mp_face_mesh.FACEMESH_LIPS))),
        'CONTOURS': list(set(itertools.chain(*self.mp_face_mesh.FACEMESH_CONTOURS))),
    }
    
    def extract_features(self, frame_rgb):
        """Extract geometric features from facial regions."""
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame_rgb.shape[:2]
        
        all_points = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks])
        
        face_center = all_points.mean(axis=0)
        face_scale = all_points.std()
        
        features = []
        
        for region_name, indices in self.REGION_LANDMARKS.items():
            region_points = all_points[indices]
            normalized_points = (region_points - face_center) / (face_scale + 1e-6)
            
            centroid = normalized_points.mean(axis=0)
            spread = normalized_points.std(axis=0)
            
            x_range = normalized_points[:, 0].max() - normalized_points[:, 0].min()
            y_range = normalized_points[:, 1].max() - normalized_points[:, 1].min()
            aspect_ratio = x_range / (y_range + 1e-6)
            
            if len(region_points) >= 3:
                try:
                    hull = cv2.convexHull(region_points[:, :2].astype(np.float32))
                    area = cv2.contourArea(hull) / (face_scale ** 2 + 1e-6)
                except:
                    area = x_range * y_range
            else:
                area = x_range * y_range
            
            features.extend([
                *centroid, *spread, x_range, y_range, aspect_ratio, area
            ])
        
        # Inter-region features
        left_eye_center = all_points[self.REGION_LANDMARKS['LEFT_EYE']].mean(axis=0)
        right_eye_center = all_points[self.REGION_LANDMARKS['RIGHT_EYE']].mean(axis=0)
        eye_distance = np.linalg.norm(left_eye_center - right_eye_center) / (face_scale + 1e-6)
        
        mouth_points = all_points[self.REGION_LANDMARKS['LIPS']]
        mouth_height = (mouth_points[:, 1].max() - mouth_points[:, 1].min()) / (face_scale + 1e-6)
        mouth_width = (mouth_points[:, 0].max() - mouth_points[:, 0].min()) / (face_scale + 1e-6)
        mouth_ratio = mouth_height / (mouth_width + 1e-6)
        
        left_brow_center = all_points[self.REGION_LANDMARKS['LEFT_EYEBROW']].mean(axis=0)
        right_brow_center = all_points[self.REGION_LANDMARKS['RIGHT_EYEBROW']].mean(axis=0)
        left_brow_raise = (left_eye_center[1] - left_brow_center[1]) / (face_scale + 1e-6)
        right_brow_raise = (right_eye_center[1] - right_brow_center[1]) / (face_scale + 1e-6)
        
        features.extend([
            eye_distance, mouth_height, mouth_width, mouth_ratio,
            left_brow_raise, right_brow_raise
        ])
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_count(self):
        return len(self.REGION_LANDMARKS) * 10 + 6
    
    def close(self):
        self.face_mesh.close()


class Region_Net(nn.Module):
    def __init__(self, input_size=66, num_classes=7):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)
        
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


class RegionDataset(torch.utils.data.Dataset):
    """Dataset for region-based features."""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]