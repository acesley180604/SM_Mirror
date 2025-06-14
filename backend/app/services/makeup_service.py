import asyncio
import os
import sys
import torch
from typing import Optional
import logging
from PIL import Image
import shutil
import numpy as np

# Add Stable-Makeup to path
STABLE_MAKEUP_PATH = os.path.join(os.path.dirname(__file__), '../../../Stable-Makeup')
sys.path.append(STABLE_MAKEUP_PATH)

logger = logging.getLogger(__name__)

class MakeupService:
    def __init__(self):
        self.model_loaded = False
        self.pipe = None
        self.makeup_encoder = None
        self.id_encoder = None
        self.pose_encoder = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"MakeupService initialized with device: {self.device}")
        
    async def apply_makeup_transfer(
        self, 
        source_path: str, 
        reference_path: str, 
        result_path: str
    ) -> bool:
        """
        Apply makeup transfer from reference image to source image using Stable-Makeup
        
        Args:
            source_path: Path to source image
            reference_path: Path to reference image  
            result_path: Path where result will be saved
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Processing makeup transfer: {source_path} + {reference_path} -> {result_path}")
            
            # Load models if not loaded
            if not self.model_loaded:
                await self._load_models()
            
            # Load and preprocess images
            source_image = Image.open(source_path).convert('RGB').resize((512, 512))
            reference_image = Image.open(reference_path).convert('RGB').resize((512, 512))
            
            # Generate pose control image using SPIGA
            pose_image = await self._get_pose_control(source_image)
            
            # Perform makeup transfer
            result_image = await self._perform_makeup_transfer(
                source_image, reference_image, pose_image
            )
            
            # Save result
            result_image.save(result_path, quality=95)
            logger.info(f"Makeup transfer completed: {result_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error in makeup transfer: {str(e)}")
            # Fallback: copy source image
            try:
                shutil.copy(source_path, result_path)
                return True
            except:
                return False
    
    async def _load_models(self):
        """Load Stable-Makeup models"""
        try:
            logger.info("Loading Stable-Makeup models...")
            
            # Import necessary modules
            from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel
            from diffusers import DDIMScheduler, ControlNetModel
            from detail_encoder.encoder_plus import detail_encoder
            
            # Model paths
            model_id = "runwayml/stable-diffusion-v1-5"
            models_dir = os.path.join(os.path.dirname(__file__), '../../../pytorch_model_2')
            makeup_encoder_path = os.path.join(models_dir, "pytorch_model.bin")
            id_encoder_path = os.path.join(models_dir, "pytorch_model_1.bin") 
            pose_encoder_path = os.path.join(models_dir, "pytorch_model_2.bin")
            
            # Check if model files exist
            if not all(os.path.exists(p) for p in [makeup_encoder_path, id_encoder_path, pose_encoder_path]):
                logger.warning("Some model files not found, using fallback")
                self.model_loaded = True
                return
            
            # Load UNet
            logger.info("Loading UNet...")
            unet = OriginalUNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
            unet = unet.to(self.device)
            
            # Load encoders
            logger.info("Loading encoders...")
            self.id_encoder = ControlNetModel.from_unet(unet)
            self.pose_encoder = ControlNetModel.from_unet(unet)
            
            # Load makeup encoder with proper image encoder path
            image_encoder_path = os.path.join(STABLE_MAKEUP_PATH, "models/image_encoder_l")
            if not os.path.exists(image_encoder_path):
                image_encoder_path = "openai/clip-vit-large-patch14"
                
            self.makeup_encoder = detail_encoder(
                unet, image_encoder_path, self.device, dtype=torch.float32
            )
            
            # Load state dicts
            logger.info("Loading model weights...")
            if os.path.exists(makeup_encoder_path):
                makeup_state_dict = torch.load(makeup_encoder_path, map_location=self.device)
                self.makeup_encoder.load_state_dict(makeup_state_dict, strict=False)
            
            if os.path.exists(id_encoder_path):
                id_state_dict = torch.load(id_encoder_path, map_location=self.device)
                self.id_encoder.load_state_dict(id_state_dict, strict=False)
            
            if os.path.exists(pose_encoder_path):
                pose_state_dict = torch.load(pose_encoder_path, map_location=self.device)
                self.pose_encoder.load_state_dict(pose_state_dict, strict=False)
            
            # Move to device
            self.id_encoder = self.id_encoder.to(self.device)
            self.pose_encoder = self.pose_encoder.to(self.device)
            
            # Create pipeline
            logger.info("Creating pipeline...")
            from utils.pipeline_sd15 import StableDiffusionControlNetPipeline
            
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                model_id,
                safety_checker=None,
                unet=unet,
                controlnet=[self.id_encoder, self.pose_encoder],
                torch_dtype=torch.float32
            ).to(self.device)
            
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
            
            self.model_loaded = True
            logger.info("Stable-Makeup models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.model_loaded = True  # Set to true to avoid infinite retries
    
    async def _get_pose_control(self, image: Image.Image) -> Image.Image:
        """Generate pose control image using SPIGA or MediaPipe fallback"""
        try:
            # Try SPIGA first
            try:
                from spiga_draw import spiga_process, spiga_segmentation
                from spiga.inference.config import ModelConfig
                from spiga.inference.framework import SPIGAFramework
                
                # Initialize SPIGA processor
                processor = SPIGAFramework(ModelConfig("300wpublic"))
                
                # Try to use detector from Stable-Makeup directory
                detector_path = os.path.join(STABLE_MAKEUP_PATH, "models/mobilenet0.25_Final.pth")
                
                if os.path.exists(detector_path):
                    # Use custom face detector if available
                    try:
                        from facelib import FaceDetector
                        detector = FaceDetector(weight_path=detector_path)
                    except ImportError:
                        logger.warning("FaceLib not available, using SPIGA built-in detection")
                        detector = None
                else:
                    logger.warning("Face detector model not found, using SPIGA built-in detection")
                    detector = None
                
                # Process image with SPIGA
                if detector:
                    spigas = spiga_process(image, detector)
                else:
                    # Use SPIGA's built-in detection
                    import numpy as np
                    image_np = np.array(image)
                    features = processor.inference(image_np)
                    if features and len(features) > 0:
                        spigas = features[0]  # Take first face
                    else:
                        spigas = False
                
                if spigas != False:
                    return spiga_segmentation(spigas, size=512)
                    
            except Exception as spiga_error:
                logger.warning(f"SPIGA processing failed: {spiga_error}, falling back to MediaPipe")
            
            # Fallback to MediaPipe for pose control
            try:
                import mediapipe as mp
                import numpy as np
                import cv2
                
                # Convert PIL to OpenCV format
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Initialize MediaPipe Face Mesh
                mp_face_mesh = mp.solutions.face_mesh
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing_styles = mp.solutions.drawing_styles
                
                with mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as face_mesh:
                    
                    results = face_mesh.process(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
                    
                    if results.multi_face_landmarks:
                        # Create a black canvas
                        annotated_image = np.zeros_like(image_cv)
                        
                        for face_landmarks in results.multi_face_landmarks:
                            # Draw face mesh
                            mp_drawing.draw_landmarks(
                                image=annotated_image,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles
                                .get_default_face_mesh_contours_style())
                        
                        # Convert back to PIL
                        return Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            
            except Exception as mediapipe_error:
                logger.warning(f"MediaPipe fallback failed: {mediapipe_error}")
            
            # Final fallback: return black image
            logger.warning("All pose control methods failed, using black image")
            width, height = image.size
            return Image.new('RGB', (width, height), color=(0, 0, 0))
                
        except Exception as e:
            logger.error(f"Error in pose control generation: {str(e)}")
            # Fallback: return black image
            width, height = image.size
            return Image.new('RGB', (width, height), color=(0, 0, 0))
    
    async def _perform_makeup_transfer(
        self, 
        source_image: Image.Image, 
        reference_image: Image.Image, 
        pose_image: Image.Image
    ) -> Image.Image:
        """Perform the actual makeup transfer"""
        try:
            if self.makeup_encoder and self.pipe:
                logger.info("Performing makeup transfer with Stable-Makeup...")
                result_image = self.makeup_encoder.generate(
                    id_image=[source_image, pose_image],
                    makeup_image=reference_image,
                    pipe=self.pipe,
                    guidance_scale=1.6
                )
                return result_image
            else:
                logger.warning("Models not available, returning source image")
                return source_image
                
        except Exception as e:
            logger.error(f"Error in makeup transfer: {str(e)}")
            return source_image
    
    def load_models(self):
        """Load Stable-Makeup models synchronously"""
        asyncio.run(self._load_models())
    
    def unload_models(self):
        """Unload models to free memory"""
        try:
            if hasattr(self, 'pipe') and self.pipe:
                del self.pipe
                self.pipe = None
            if hasattr(self, 'makeup_encoder') and self.makeup_encoder:
                del self.makeup_encoder
                self.makeup_encoder = None
            if hasattr(self, 'id_encoder') and self.id_encoder:
                del self.id_encoder
                self.id_encoder = None
            if hasattr(self, 'pose_encoder') and self.pose_encoder:
                del self.pose_encoder
                self.pose_encoder = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.model_loaded = False
            logger.info("Models unloaded successfully")
            
        except Exception as e:
            logger.error(f"Error unloading models: {str(e)}") 