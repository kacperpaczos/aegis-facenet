from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import numpy as np
import base64
import cv2
from aiohttp import web
import logging
import io
import json
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FacenetServer:
    def __init__(self, host="0.0.0.0", port=8081, debug=True):
        self.host = host
        self.port = port
        self.app = web.Application(client_max_size=1024*1024*50)
        self.debug = debug
        self.recognition_system = PersonRecognitionSystem(debug=debug)
        self.setup_routes()
        if self.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("FacenetServer initialized in debug mode")

    def setup_routes(self):
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_post('/process_frame', self.process_frame)
        self.app.router.add_post('/train', self.train_model)
        self.app.router.add_post('/process_frames', self.process_frames)

    async def health_check(self, request):
        return web.Response(text='{"status": "healthy"}', content_type='application/json')

    async def process_frame(self, request):
        if self.debug:
            logger.debug("=== Frame Processing Started ===")
        try:
            data = await request.json()
            if self.debug:
                logger.debug(f"Received frame data of size: {len(data['frame'])} bytes")
            
            frame_bytes = base64.b64decode(data['frame'])
            if self.debug:
                logger.debug("Base64 decoding completed successfully")
            
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.debug:
                logger.debug(f"Frame converted to RGB format (shape: {frame_rgb.shape})")
        
            embedding = self.recognition_system.get_embedding(frame_rgb)
            if embedding is not None:
                if self.debug:
                    logger.debug("Face embedding generated successfully")
                similarity = self.recognition_system.compare_embeddings(embedding)
                if self.debug:
                    logger.debug(f"Similarity score with reference: {similarity:.4f}")
                    logger.debug("=== Frame Processing Completed: Face Found ===")
                return web.Response(text=json.dumps({
                    "match_found": similarity > 0.7,
                    "similarity": float(similarity),
                    "face_detected": True,
                    "warning": "No reference pattern" if self.recognition_system.current_person_embedding is None else None
                }), content_type='application/json')
        
            if self.debug:
                logger.debug("=== Frame Processing Completed: No Face Detected ===")
            return web.Response(text=json.dumps({
                "match_found": False,
                "similarity": 0.0,
                "face_detected": False,
                "warning": "No face detected"
            }), content_type='application/json')

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return web.Response(text=json.dumps({
                "status": "error",
                "face_detected": False,
                "message": str(e)
            }), content_type='application/json')

    async def train_model(self, request):
        if self.debug:
            logger.debug("=== Model Training Started ===")
            logger.debug("=== Configuration ===")
            logger.debug("- MTCNN model: Enabled (keep_all=True)")
            logger.debug("- ResNet model: InceptionResnetV1 (pretrained=vggface2)")
        try:
            data = await request.json()
            video_path = data.get('video_path')
            if not video_path:
                raise ValueError("Video path not provided")
            
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"File not found: {video_path}")
            
            if self.debug:
                logger.debug("=== Video Analysis ===")
                logger.debug(f"Processing video file: {video_path}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Cannot open video file")
            
            if self.debug:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = total_frames / fps if fps > 0 else 0
                resolution = (
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                )
                logger.debug("Video properties:")
                logger.debug(f"- Resolution: {resolution[0]}x{resolution[1]}")
                logger.debug(f"- FPS: {fps:.2f}")
                logger.debug(f"- Total frames: {total_frames}")
                logger.debug(f"- Duration: {duration:.2f} seconds")
                logger.debug("=== Starting Frame Analysis ===")
            
            best_embedding = None
            best_confidence = 0.0
            frame_count = 0
            faces_detected = 0
            face_probabilities = []
            face_sizes = []
            processing_times = []
            
            while True:
                start_time = datetime.now()
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if self.debug and frame_count % 10 == 0:
                    progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                    logger.debug(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
                    logger.debug(f"Faces detected so far: {faces_detected}")
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self.recognition_system.mtcnn(frame_rgb)
                
                if faces is not None:
                    faces_detected += 1
                    boxes, probs = self.recognition_system.mtcnn.detect(frame_rgb)
                    
                    if probs is not None and boxes is not None:
                        face_probabilities.extend([float(p) for p in probs])
                        # Calculate face sizes as percentage of frame
                        for box in boxes:
                            face_width = box[2] - box[0]
                            face_height = box[3] - box[1]
                            face_size = (face_width * face_height) / (frame.shape[0] * frame.shape[1]) * 100
                            face_sizes.append(face_size)
                    
                    embedding = self.recognition_system.resnet(faces)
                    confidence = torch.max(faces[0].detach()).item()
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_embedding = embedding
                        if self.debug:
                            logger.debug("=== New Best Embedding Found ===")
                            logger.debug(f"- Confidence: {confidence:.4f}")
                            logger.debug(f"- Face probability: {max(probs) if probs is not None else 'N/A'}")
                            if boxes is not None:
                                box = boxes[0]
                                logger.debug(f"- Face position: x1={box[0]:.0f}, y1={box[1]:.0f}, "
                                           f"x2={box[2]:.0f}, y2={box[3]:.0f}")
                                logger.debug(f"- Face size: {face_sizes[-1]:.2f}% of frame")
                
                processing_times.append((datetime.now() - start_time).total_seconds())
            
            cap.release()
            
            if best_embedding is not None:
                self.recognition_system.current_person_embedding = best_embedding
                
                avg_prob = sum(face_probabilities) / len(face_probabilities) if face_probabilities else 0
                avg_size = sum(face_sizes) / len(face_sizes) if face_sizes else 0
                avg_time = sum(processing_times) / len(processing_times)
                
                if self.debug:
                    logger.debug("\n=== Training Completed Successfully ===")
                    logger.debug("\nPerformance Statistics:")
                    logger.debug(f"- Average processing time per frame: {avg_time*1000:.2f}ms")
                    logger.debug(f"- Processing rate: {1/avg_time:.2f} FPS")
                    
                    logger.debug("\nFace Detection Statistics:")
                    logger.debug(f"- Total frames analyzed: {frame_count}")
                    logger.debug(f"- Total faces detected: {faces_detected}")
                    logger.debug(f"- Face detection rate: {(faces_detected/frame_count*100):.2f}%")
                    logger.debug(f"- Average face size: {avg_size:.2f}% of frame")
                    logger.debug(f"- Face size range: {min(face_sizes):.2f}% - {max(face_sizes):.2f}%")
                    
                    logger.debug("\nQuality Metrics:")
                    logger.debug(f"- Average probability: {avg_prob:.4f}")
                    logger.debug(f"- Maximum probability: {max(face_probabilities):.4f}")
                    logger.debug(f"- Minimum probability: {min(face_probabilities):.4f}")
                    logger.debug(f"- Best embedding confidence: {best_confidence:.4f}")
                    logger.debug("\n=== Training Session Complete ===")
                
                return web.Response(text=json.dumps({
                    "status": "success",
                    "message": "Model trained successfully",
                    "statistics": {
                        "frames_processed": frame_count,
                        "faces_detected": faces_detected,
                        "face_detection_rate": faces_detected / frame_count if frame_count > 0 else 0,
                        "avg_probability": float(avg_prob),
                        "max_probability": float(max(face_probabilities)),
                        "min_probability": float(min(face_probabilities)),
                        "best_confidence": float(best_confidence),
                        "avg_face_size": float(avg_size),
                        "processing_fps": float(1/avg_time)
                    }
                }), content_type='application/json')
            else:
                if self.debug:
                    logger.debug("=== Training Completed: No Suitable Face Found ===")
                return web.Response(text=json.dumps({
                    "status": "error",
                    "message": "No suitable face found in the video"
                }), content_type='application/json')

        except Exception as e:
            logger.error(f"Error during model training: {e}")
            return web.Response(text=json.dumps({
                "status": "error",
                "message": str(e)
            }), content_type='application/json')

    async def process_frames(self, request):
        if self.debug:
            logger.debug("=== Batch Frame Processing Started ===")
        try:
            data = await request.json()
            frames_data = data['frames']
            if self.debug:
                logger.debug(f"Received {len(frames_data)} frames for processing")
            
            best_match = {"match_found": False, "similarity": 0.0, "face_detected": False}
            
            for i, frame_bytes in enumerate(frames_data, 1):
                if self.debug:
                    logger.debug(f"Processing frame {i}/{len(frames_data)} - "
                               f"Current best similarity: {best_match['similarity']:.4f}")
                    
                frame_data = base64.b64decode(frame_bytes)
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                embedding = self.recognition_system.get_embedding(frame_rgb)
                if embedding is not None:
                    similarity = self.recognition_system.compare_embeddings(embedding)
                    current_result = {
                        "match_found": similarity > 0.7,
                        "similarity": float(similarity),
                        "face_detected": True
                    }
                    
                    if current_result["similarity"] > best_match["similarity"]:
                        best_match = current_result
                        if self.debug:
                            logger.debug(f"Found better match - Similarity: {similarity:.4f}")
                    
                    if best_match["match_found"]:
                        if self.debug:
                            logger.debug("Sufficient match found - stopping further processing")
                        break
            
            if self.debug:
                logger.debug(f"=== Batch Processing Completed: {best_match} ===")
            
            return web.Response(text=json.dumps({
                **best_match,
                "warning": "No reference pattern" if self.recognition_system.current_person_embedding is None else None
            }), content_type='application/json')

        except Exception as e:
            logger.error(f"Error processing frames batch: {e}")
            return web.Response(text=json.dumps({
                "status": "error",
                "face_detected": False,
                "message": str(e)
            }), content_type='application/json')

    def run(self):
        web.run_app(self.app, host=self.host, port=self.port)

class PersonRecognitionSystem:
    def __init__(self, debug=True):
        self.reset_model()
        self.debug = debug
        
    def reset_model(self):
        self.mtcnn = MTCNN(keep_all=True)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.current_person_embedding = None

    def get_embedding(self, frame):
        try:
            if self.debug:
                logger.debug(f"Attempting face detection on frame of shape: {frame.shape}")
            
            faces = self.mtcnn(frame)
            if faces is None:
                logger.warning("No faces detected in frame")
                return None
                
            if self.debug:
                logger.debug(f"Detected {len(faces)} faces")
                logger.debug(f"Face tensor shape: {faces.shape}")
            
            embedding = self.resnet(faces)
            if self.debug:
                logger.debug(f"Generated embedding of shape: {embedding.shape}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None

    def compare_embeddings(self, embedding):
        if self.current_person_embedding is None:
            logger.warning("No reference pattern available - train the model first")
            return 0.0
            
        current_embedding = self.current_person_embedding.flatten()
        new_embedding = embedding.flatten()
        
        similarity = torch.nn.functional.cosine_similarity(
            current_embedding.unsqueeze(0),
            new_embedding.unsqueeze(0)
        ).item()
        
        return similarity

if __name__ == "__main__":
    server = FacenetServer()
    server.run()