"""Enhanced camera feed with depth, landmarks, face recognition, and liveness detection."""
import logging
from camera import RealSenseCamera
from encoder import FaceEncoder
from database import FaceDatabase
from liveness import LivenessDetector
import face_recognition
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)

def show_enhanced_camera_feed():
    print("Starting enhanced camera feed... Press 'q' to quit, 's' to save face, 'c' to clear")
    
    try:
        camera = RealSenseCamera()
        encoder = FaceEncoder(detection_model="cnn", encoding_model="large", num_jitters=3)
        db = FaceDatabase()
        liveness = LivenessDetector()
        
        face_counter = 0
        
        while True:
            frame = camera.read()
            if frame is None:
                print("No frame captured")
                continue
            
            h, w = frame.color_bgr.shape[:2]
            
            # Convert BGR to RGB for face_recognition
            frame_rgb = cv2.cvtColor(frame.color_bgr, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = encoder.detect(frame.color_bgr)
            
            # Get landmarks for drawing (landmarks useful for visual debug)
            face_landmarks_list = face_recognition.face_landmarks(frame_rgb)
            
            # Create display frame
            display_frame = frame.color_bgr.copy()
            
            # Draw depth map as heatmap (small window in corner)
            depth_normalized = np.clip(frame.depth_z16 / 5000.0, 0, 1)  # Normalize to 5m range
            depth_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            depth_resized = cv2.resize(depth_colored, (200, 150))
            display_frame[10:160, 10:210] = depth_resized
            
            # Add depth label
            cv2.putText(display_frame, "Depth (0-5m)", (15, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Head pose/human detection removed for lightweight build
            
            # Process each face
            for idx, (top, right, bottom, left) in enumerate(face_locations):
                # Check liveness first
                is_alive, liveness_conf, liveness_msg = liveness.is_alive(frame.depth_z16, (top, right, bottom, left), frame.depth_scale)
                
                # Draw face bounding box
                box_color = (0, 255, 0) if is_alive else (0, 0, 255)  # Green=live, Red=spoofed
                cv2.rectangle(display_frame, (left, top), (right, bottom), box_color, 3)
                
                # Get embedding
                embedding = encoder.encode(frame.color_bgr, (top, right, bottom, left))
                
                if embedding is not None and is_alive:
                    # Try to match against database (only if face is alive)
                    name, distance = db.match(embedding, tolerance=0.45)
                    
                    if name:
                        label = f"{name} ({distance:.2f})"
                        color = (0, 255, 0)  # Green for match
                    else:
                        label = "Unknown (Live)"
                        color = (0, 255, 255)  # Yellow for unknown but live
                    
                    # Draw label
                    cv2.putText(display_frame, label, (left, top - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Draw liveness info
                    cv2.putText(display_frame, f"✓ LIVE ({liveness_conf:.0%})", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Draw face info box
                    info_y = h - 120
                    cv2.rectangle(display_frame, (10, info_y), (450, h - 10), (0, 0, 0), -1)
                    cv2.putText(display_frame, f"Face #{idx + 1}: {label}", (20, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    cv2.putText(display_frame, f"Status: LIVE (Liveness: {liveness_conf:.0%})", (20, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    cv2.putText(display_frame, f"DB Size: {len(db.names)} faces", (20, info_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                else:
                    # Face is spoofed or embedding failed
                    label = "SPOOFED (Photo/Screen)"
                    cv2.putText(display_frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # Draw face info box
                    info_y = h - 100
                    cv2.rectangle(display_frame, (10, info_y), (450, h - 10), (0, 0, 0), -1)
                    cv2.putText(display_frame, f"Face #{idx + 1}: {label}", (20, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.putText(display_frame, liveness_msg, (20, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    # Human detection removed
            
            # Draw landmarks as circles
            for face_landmarks in face_landmarks_list:
                for feature_name, points in face_landmarks.items():
                    for x, y in points:
                        cv2.circle(display_frame, (int(x), int(y)), 2, (0, 0, 255), -1)
            
            # Draw stats
            stats_text = f"Faces: {len(face_locations)}"
            cv2.putText(display_frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("Enhanced Camera Feed (q=quit, s=save, c=clear)", display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and len(face_locations) > 0:
                # Save first live face only if liveness passes
                is_alive, _, _ = liveness.is_alive(frame.depth_z16, face_locations[0], frame.depth_scale)
                if is_alive:
                    embedding = encoder.encode(frame.color_bgr, face_locations[0])
                    if embedding is not None:
                        face_counter += 1
                        name = f"Person_{face_counter}"
                        db.add(name, embedding)
                        print(f"✓ Saved face as: {name}")
                else:
                    print("✗ Cannot save spoofed face (not a real person)")
            elif key == ord('s'):
                print("✗ Cannot save: No live face detected")
            elif key == ord('c'):
                db.clear()
                face_counter = 0
                print("✓ Database cleared")
        
        cv2.destroyAllWindows()
        camera.release()
        print("Camera feed closed")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    show_enhanced_camera_feed()
