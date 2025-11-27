import cv2
import torch
import numpy as np
import time
from collections import deque

from cnn import Conv_Net
from ffn import FF_Net
from region import Region_Net, RegionFeatureExtractor

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

COLORS = {
    'CNN': (255, 100, 100),
    'FFN': (100, 255, 100),
    'Region': (100, 100, 255),
    'box': (0, 255, 0),
}


class EmotionDetector:
    def __init__(self, cnn_path='emotion_cnn_best.pth', 
                 ffn_path='emotion_ffn_best.pth',
                 region_path='emotion_region_best.pth'):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # Load CNN
        self.cnn = Conv_Net()
        try:
            self.cnn.load_state_dict(torch.load(cnn_path, map_location=self.device))
            self.cnn = self.cnn.to(self.device).eval()
            self.cnn_available = True
            print("CNN loaded")
        except FileNotFoundError:
            print(f"CNN not found: {cnn_path}")
            self.cnn_available = False
        
        # Load FFN
        self.ffn = FF_Net()
        try:
            self.ffn.load_state_dict(torch.load(ffn_path, map_location=self.device))
            self.ffn = self.ffn.to(self.device).eval()
            self.ffn_available = True
            print("FFN loaded")
        except FileNotFoundError:
            print(f"FFN not found: {ffn_path}")
            self.ffn_available = False
        
        # Load Region
        self.region_net = Region_Net()
        try:
            self.region_net.load_state_dict(torch.load(region_path, map_location=self.device))
            self.region_net = self.region_net.to(self.device).eval()
            self.region_extractor = RegionFeatureExtractor(static_mode=False)
            self.region_available = True
            print("Region loaded")
        except FileNotFoundError:
            print(f"Region not found: {region_path}")
            self.region_available = False
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        self.history = {k: deque(maxlen=5) for k in ['CNN', 'FFN', 'Region']}
        self.fps_history = deque(maxlen=30)
    
    def preprocess(self, face_roi):
        """Convert to 48x48 grayscale tensor."""
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        tensor = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)
    
    def smooth(self, model_name, probs):
        """Smooth predictions over time."""
        self.history[model_name].append(probs)
        return np.mean(list(self.history[model_name]), axis=0)
    
    def predict(self, frame, face_roi):
        """Run all models and return predictions."""
        results = {}
        tensor = self.preprocess(face_roi)
        
        with torch.no_grad():
            if self.cnn_available:
                out = torch.softmax(self.cnn(tensor), dim=1).cpu().numpy()[0]
                probs = self.smooth('CNN', out)
                results['CNN'] = {'emotion': EMOTIONS[np.argmax(probs)], 
                                 'confidence': np.max(probs), 'probs': probs}
            
            if self.ffn_available:
                out = torch.softmax(self.ffn(tensor.view(-1, 2304)), dim=1).cpu().numpy()[0]
                probs = self.smooth('FFN', out)
                results['FFN'] = {'emotion': EMOTIONS[np.argmax(probs)], 
                                 'confidence': np.max(probs), 'probs': probs}
            
            if self.region_available:
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                features = self.region_extractor.extract_features(face_rgb)
                
                if features is not None:
                    feat_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                    out = torch.softmax(self.region_net(feat_tensor), dim=1).cpu().numpy()[0]
                    probs = self.smooth('Region', out)
                    results['Region'] = {'emotion': EMOTIONS[np.argmax(probs)], 
                                        'confidence': np.max(probs), 'probs': probs}
                else:
                    results['Region'] = {'emotion': 'No face', 'confidence': 0, 'probs': None}
        
        return results
    
    def draw_results(self, frame, results, x, y, w, h, show_probs=False):
        """Draw bounding box and predictions."""
        cv2.rectangle(frame, (x, y), (x + w, y + h), COLORS['box'], 2)
        
        y_offset = y - 10
        for model_name, pred in results.items():
            text = f"{model_name}: {pred['emotion']} ({pred['confidence']:.0%})"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y_offset - th - 5), (x + tw + 10, y_offset + 5), (0,0,0), -1)
            cv2.putText(frame, text, (x + 5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[model_name], 2)
            y_offset -= 30
        
        if show_probs:
            self.draw_prob_bars(frame, results)
        
        return frame
    
    def draw_prob_bars(self, frame, results):
        """Draw probability bars."""
        bar_h, bar_w, start_y = 12, 120, 50
        
        for idx, (name, pred) in enumerate(results.items()):
            if pred['probs'] is None:
                continue
            
            start_x = 10 + idx * (bar_w + 30)
            cv2.putText(frame, name, (start_x, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[name], 1)
            
            for i, (emo, prob) in enumerate(zip(EMOTIONS, pred['probs'])):
                y_pos = start_y + i * (bar_h + 2)
                cv2.rectangle(frame, (start_x, y_pos), (start_x + bar_w, y_pos + bar_h), (50,50,50), -1)
                cv2.rectangle(frame, (start_x, y_pos), (start_x + int(bar_w * prob), y_pos + bar_h),
                            COLORS[name] if prob == max(pred['probs']) else (100,100,100), -1)
                cv2.putText(frame, emo[:3], (start_x + bar_w + 3, y_pos + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
    
    def run(self, camera_id=0, show_probs=True, show_landmarks=False):
        """Main loop."""
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("Cannot open camera")
            return
        
        print("\nControls: q=quit, p=toggle probs, l=toggle landmarks, s=screenshot")
        
        while True:
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
            
            for (x, y, w, h) in faces:
                pad = int(0.1 * w)
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
                face_roi = frame[y1:y2, x1:x2]
                
                if face_roi.size == 0:
                    continue
                
                results = self.predict(frame, face_roi)
                frame = self.draw_results(frame, results, x, y, w, h, show_probs)
            
            if show_landmarks and self.region_available:
                frame = self.region_extractor.draw_landmarks(frame)
            
            fps = 1.0 / (time.time() - start + 1e-6)
            self.fps_history.append(fps)
            cv2.putText(frame, f"FPS: {np.mean(self.fps_history):.1f}", 
                       (frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            
            cv2.imshow('Emotion Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                show_probs = not show_probs
            elif key == ord('l'):
                show_landmarks = not show_landmarks
            elif key == ord('s'):
                cv2.imwrite(f"screenshot_{int(time.time())}.png", frame)
                print("Screenshot saved")
        
        cap.release()
        cv2.destroyAllWindows()
        if self.region_available:
            self.region_extractor.close()


def main():
    import argparse
    
    detector = EmotionDetector('emotion_cnn_best.pth', 'emotion_ffn_best.pth', 'emotion_region_best.pth')
    detector.run(0, True, False)


if __name__ == '__main__':
    main()