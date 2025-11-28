import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import mediapipe as mp

class RegionFeatureExtractor:    
    REGION_LANDMARKS = {
        'left_eye': [33, 160, 158, 133, 153, 144, 163, 7],
        'right_eye': [362, 385, 387, 263, 373, 380, 390, 249],
        'left_eyebrow': [70, 63, 105, 66, 107, 55, 65],
        'right_eyebrow': [300, 293, 334, 296, 336, 285, 295],
        'nose': [1, 2, 98, 327, 168, 6, 197, 195, 5],
        'mouth': [61, 291, 0, 17, 13, 14, 78, 308, 415, 310, 311, 312, 82, 87]
    }
    
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
            
            rel_position = centroid
            
            features.extend([
                *centroid, *spread, x_range, y_range, aspect_ratio, area, *rel_position
            ])
        
        # Inter-region features
        left_eye_center = all_points[self.REGION_LANDMARKS['left_eye']].mean(axis=0)
        right_eye_center = all_points[self.REGION_LANDMARKS['right_eye']].mean(axis=0)
        eye_distance = np.linalg.norm(left_eye_center - right_eye_center) / (face_scale + 1e-6)
        
        mouth_points = all_points[self.REGION_LANDMARKS['mouth']]
        mouth_height = (mouth_points[:, 1].max() - mouth_points[:, 1].min()) / (face_scale + 1e-6)
        mouth_width = (mouth_points[:, 0].max() - mouth_points[:, 0].min()) / (face_scale + 1e-6)
        mouth_ratio = mouth_height / (mouth_width + 1e-6)
        
        left_brow_center = all_points[self.REGION_LANDMARKS['left_eyebrow']].mean(axis=0)
        right_brow_center = all_points[self.REGION_LANDMARKS['right_eyebrow']].mean(axis=0)
        left_brow_raise = (left_eye_center[1] - left_brow_center[1]) / (face_scale + 1e-6)
        right_brow_raise = (right_eye_center[1] - right_brow_center[1]) / (face_scale + 1e-6)
        
        features.extend([
            eye_distance, mouth_height, mouth_width, mouth_ratio,
            left_brow_raise, right_brow_raise
        ])
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_count(self):
        return len(self.REGION_LANDMARKS) * 13 + 6
    
    def draw_landmarks(self, frame, draw_regions=True):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                if draw_regions:
                    h, w = frame.shape[:2]
                    colors = {
                        'left_eye': (255, 0, 0), 'right_eye': (255, 0, 0),
                        'left_eyebrow': (0, 255, 0), 'right_eyebrow': (0, 255, 0),
                        'nose': (0, 0, 255), 'mouth': (255, 255, 0)
                    }
                    
                    for region_name, indices in self.REGION_LANDMARKS.items():
                        for idx in indices:
                            lm = face_landmarks.landmark[idx]
                            x, y = int(lm.x * w), int(lm.y * h)
                            cv2.circle(frame, (x, y), 2, colors[region_name], -1)
        
        return frame
    
    def close(self):
        self.face_mesh.close()


class Region_Net(nn.Module):
    def __init__(self, input_size=84, num_classes=7):
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