import datetime
import json
import logging
import os
import shutil
import sys # For potentially adding AdaFace repo path
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict

import faiss
import numpy as np
from PIL import Image, UnidentifiedImageError
import cv2 # For image processing and alignment
import torch

# --- Face Detector: Using MTCNN from facenet-pytorch ---
try:
    from facenet_pytorch import MTCNN
    logging.info("Successfully imported MTCNN from facenet-pytorch for FAISS builder.")
except ImportError:
    logging.critical("CRITICAL: 'facenet-pytorch' library not found. Please install it: pip install facenet-pytorch")
    raise
# --- End Face Detector Import ---

# --- CRITICAL: Import 'net' from AdaFace GitHub Repository ---
try:
    import net # This should import the net.py from the AdaFace repository
    logging.info(f"Successfully imported 'net.py' for AdaFace model definitions (FAISS builder).")
except ImportError as e:
    logging.critical(f"CRITICAL IMPORT ERROR (FAISS Builder): Could not import 'net.py': {e}. "
                     "Ensure 'net.py' from 'github.com/mk-minchul/AdaFace/' is in your Python path.")
    raise
# --- End Critical Import Section ---

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("adaface_faiss_builder_mtcnn.log", mode='w') # Updated log file name
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration Class ---
class AdaFaceFaissBuilderConfig:
    BASE_IMAGE_SOURCE_DIR = Path("project-02/yenidenemem_few_shot/face_teslim/face_db/Human/Classes")

    OUTPUT_FAISS_DIR = Path("project-02/yenidenemem_few_shot/face_teslim/SOTA/faissdb_mtcnn_ada")
    FAISS_INDEX_FILENAME = "adaface_prototype_ir100_mtcnn.index"
    FAISS_MAP_FILENAME = "adaface_prototype_ir100_mtcnn_map.json"

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu" # For PyTorch models (AdaFace & MTCNN)

    # Face Detector: MTCNN (facenet-pytorch)
    # MTCNN thresholds are typically handled by its internal stages (P-Net, R-Net, O-Net).
    # We'll use its `min_face_size` and rely on its default confidence, or filter by probability if needed.
    FACE_DETECTOR_MIN_FACE_SIZE: int = 20 # For MTCNN: Minimum face size to detect.
    FACE_DETECTOR_CONFIDENCE_THRESHOLD: float = 0.90 # MTCNN returns a probability; we can filter on this.

    # AdaFace Backbone Embedding Extractor
    ADA_BACKBONE_ARCH: str = 'ir_101' # Corrected to match your .ckpt if it's IR-101
    ADA_BACKBONE_WEIGHTS_PATH: str = 'project-02/yenidenemem_few_shot/face_teslim/SOTA/models/adaface_ir101_webface12m.ckpt'
    EMBEDDING_DIMENSION: int = 512

    ALIGNMENT_TARGET_LANDMARKS = np.array([
        [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
    ALIGNMENT_CROP_SIZE: Tuple[int, int] = (112, 112)
    MIN_IMAGE_SIZE_TO_CONSIDER = 40 # Min width/height of original image before trying detection

    def __init__(self):
        self.OUTPUT_FAISS_DIR.mkdir(parents=True, exist_ok=True)
        self.FAISS_INDEX_PATH = self.OUTPUT_FAISS_DIR / self.FAISS_INDEX_FILENAME
        self.FAISS_MAP_PATH = self.OUTPUT_FAISS_DIR / self.FAISS_MAP_FILENAME
        
        if not self.BASE_IMAGE_SOURCE_DIR.exists() or not self.BASE_IMAGE_SOURCE_DIR.is_dir():
            msg = f"CRITICAL: `BASE_IMAGE_SOURCE_DIR` ('{self.BASE_IMAGE_SOURCE_DIR}') error."
            logger.critical(msg); raise FileNotFoundError(msg)
        if not Path(self.ADA_BACKBONE_WEIGHTS_PATH).is_file():
            msg = f"CRITICAL: `ADA_BACKBONE_WEIGHTS_PATH` ('{self.ADA_BACKBONE_WEIGHTS_PATH}') error."
            logger.critical(msg); raise FileNotFoundError(msg)

        logger.info(f"AdaFace FAISS Builder (MTCNN Detector) Configured. Source: '{self.BASE_IMAGE_SOURCE_DIR}'")
        logger.info(f"Output Index: '{self.FAISS_INDEX_PATH}'")

# --- Utility Functions ---
def normalize_vector(vec: np.ndarray) -> np.ndarray: # Unchanged
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-6 else np.zeros_like(vec).astype('float32')

def align_and_crop_face(image_np_rgb: np.ndarray, landmarks_5pts_ordered: np.ndarray,
                        target_landmarks: np.ndarray = AdaFaceFaissBuilderConfig.ALIGNMENT_TARGET_LANDMARKS,
                        crop_size: Tuple[int, int] = AdaFaceFaissBuilderConfig.ALIGNMENT_CROP_SIZE) -> Optional[np.ndarray]:
    try: # Unchanged
        transform_matrix, _ = cv2.estimateAffinePartial2D(landmarks_5pts_ordered.astype(np.float32), target_landmarks.astype(np.float32))
        if transform_matrix is None: return None
        aligned_face_rgb = cv2.warpAffine(image_np_rgb, transform_matrix, crop_size, borderValue=0.0)
        return aligned_face_rgb
    except cv2.error as e: logger.debug(f"OpenCV align error: {e}"); return None
    except Exception as e: logger.error(f"Align error: {e}", exc_info=False); return None

# --- Core Classes for FAISS Building ---
class BuilderFaceDetector: # MODIFIED to use MTCNN from facenet-pytorch
    def __init__(self, device: str, min_face_size: int, confidence_threshold: float):
        self.device = torch.device(device)
        self.min_face_size = min_face_size
        self.confidence_threshold = confidence_threshold # Will be applied to MTCNN's probability
        try:
            # image_size: expected size of the input image (can affect performance/accuracy)
            # margin: margin to add to the face crop (we handle cropping after alignment)
            # post_process=True: normalizes image tensors to [-1, 1] (we'll use raw PIL images)
            # select_largest=False: we want all faces then pick largest
            self.mtcnn = MTCNN(
                keep_all=True, # Detect all faces
                device=self.device,
                min_face_size=self.min_face_size,
                post_process=False, # We want raw landmark coordinates from PIL image
                select_largest=False # We will select largest manually if needed
            )
            logger.info(f"BuilderFaceDetector: MTCNN (facenet-pytorch) initialized on {self.device}.")
        except Exception as e:
            logger.error(f"BuilderFaceDetector: Failed to initialize MTCNN: {e}", exc_info=True)
            raise RuntimeError(f"MTCNN initialization failed for builder: {e}")

    def detect_largest_face(self, image_pil_rgb: Image.Image) -> Optional[Dict[str, Any]]:
        """
        Detects all faces using MTCNN and returns info for the largest one meeting confidence.
        Input: image_pil_rgb (PIL Image in RGB order).
        Output: Dict with 'box', 'landmarks' (ordered for alignment), 'confidence', or None.
        """
        if image_pil_rgb is None:
            logger.warning("BuilderFaceDetector received None image.")
            return None
        try:
            # MTCNN's detect method returns: boxes (Nx4), probs (N,), landmarks (Nx5x2)
            # It expects PIL image as input.
            boxes, probs, landmarks_all = self.mtcnn.detect(image_pil_rgb, landmarks=True)
            
            if boxes is None or landmarks_all is None: # No faces detected
                return None

            largest_face_info = None
            max_area = 0

            for i in range(len(boxes)):
                if probs[i] < self.confidence_threshold:
                    continue

                box = boxes[i] # [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)

                if area > max_area and x2 > x1 and y2 > y1: # Valid and largest so far
                    max_area = area
                    
                    # Landmarks from facenet-pytorch MTCNN are [left_eye, right_eye, nose, mouth_left, mouth_right]
                    # This order matches our AppConfig.ALIGNMENT_TARGET_LANDMARKS.
                    landmarks_np = landmarks_all[i].astype(np.int32) # Already 5x2 (x,y)
                    
                    largest_face_info = {
                        "box": [x1, y1, x2, y2],
                        "landmarks": landmarks_np, # Already in the correct order
                        "confidence": float(probs[i])
                    }
            return largest_face_info
            
        except Exception as e:
            logger.error(f"BuilderFaceDetector: Error during MTCNN detection: {e}", exc_info=True)
            return None

class BuilderEmbeddingExtractor: # Unchanged from your previously provided version
    def __init__(self, model_arch: str, weights_path: str, embedding_dim: int, device: str):
        self.model_arch_name=model_arch; self.weights_path=Path(weights_path)
        self.embedding_dim=embedding_dim; self.device=torch.device(device)
        self.model:Optional[torch.nn.Module]=None; self._load_model()
    def _load_model(self):
        if not self.weights_path.exists(): logger.critical(f"BuilderEE: AdaFace .ckpt NOT FOUND:{self.weights_path}");raise FileNotFoundError(f"AdaFace .ckpt missing:{self.weights_path}")
        try:
            logger.info(f"BuilderEE:Loading AdaFace arch:'{self.model_arch_name}' via net.build_model.")
            self.model=net.build_model(self.model_arch_name)
            ckpt=torch.load(self.weights_path,map_location=torch.device('cpu'))
            s_dict=ckpt.get('state_dict');
            if not s_dict:raise KeyError("'state_dict' missing in .ckpt.")
            model_s_dict={k[6:]:v for k,v in s_dict.items() if k.startswith('model.')}
            if not model_s_dict and s_dict:model_s_dict=s_dict
            if not model_s_dict:raise ValueError("Could not prep valid state_dict from .ckpt.")
            self.model.load_state_dict(model_s_dict,strict=True)
            self.model.to(self.device);self.model.eval()
            logger.info(f"BuilderEE:AdaFace backbone '{self.model_arch_name}' loaded from '{self.weights_path}'.")
            dummy_in=self._preprocess_input(np.zeros((112,112,3),dtype=np.uint8))
            with torch.no_grad():_,_=self.model(dummy_in.to(self.device))
        except Exception as e: logger.critical(f"BuilderEE:CRITICAL Error loading AdaFace backbone:{e}",exc_info=True); raise RuntimeError(f"Failed to load AdaFace backbone for builder:{e}")
    def _preprocess_input(self,aligned_np_rgb:np.ndarray)->torch.Tensor:
        bgr_norm=((aligned_np_rgb[:,:,::-1].astype(np.float32)/255.0)-0.5)/0.5
        return torch.tensor(bgr_norm.transpose(2,0,1)).unsqueeze(0).float()
    def extract_embedding(self,aligned_np_rgb:np.ndarray)->Optional[np.ndarray]:
        if self.model is None:return None
        try:
            tensor_in=self._preprocess_input(aligned_np_rgb).to(self.device)
            with torch.no_grad():feature,_=self.model(tensor_in)
            emb_np=feature.cpu().numpy().flatten()
            if emb_np.shape[0]!=self.embedding_dim:return None
            return normalize_vector(emb_np)
        except Exception as e:logger.warning(f"BuilderEE:Error extracting embedding:{e}",exc_info=False);return None

class FaissDBHandler: # Unchanged from your previously provided version
    def __init__(self, emb_dim: int): self.emb_dim = emb_dim
    def create_and_save_index(self, embs: np.ndarray, ids: List[str], idx_path: Path, map_path: Path) -> bool:
        if embs.ndim==1 and embs.shape[0]==0: embs=embs.reshape(0,self.emb_dim)
        if embs.shape[0]!=len(ids) or (embs.shape[0]>0 and embs.shape[1]!=self.emb_dim):
            logger.error(f"Emb/ID count or dim mismatch.Embs:{embs.shape},IDs:{len(ids)},ExpDim:{self.emb_dim}");return False
        try:
            logger.info(f"Creating FAISS index '{idx_path.name}' with {embs.shape[0]} proto vec.")
            idx=faiss.IndexFlatIP(self.emb_dim)
            if embs.shape[0]>0: idx.add(embs.astype(np.float32))
            faiss.write_index(idx,str(idx_path))
            with open(map_path,'w',encoding='utf-8') as f:json.dump(ids,f,indent=2,ensure_ascii=False)
            logger.info(f"FAISS index & map saved:'{idx_path.name}','{map_path.name}'")
            return True
        except Exception as e:logger.error(f"Err creating/saving FAISS idx:{e}",exc_info=True);return False

class AdaFaceFaissBuilder: # Main logic, adjusted for MTCNN
    def __init__(self, config: AdaFaceFaissBuilderConfig):
        self.config = config
        self.face_detector = BuilderFaceDetector( # Uses MTCNN
            device=config.DEVICE, # MTCNN can run on CUDA via PyTorch
            min_face_size=config.FACE_DETECTOR_MIN_FACE_SIZE,
            confidence_threshold=config.FACE_DETECTOR_CONFIDENCE_THRESHOLD
        )
        self.embedding_extractor = BuilderEmbeddingExtractor(
            config.ADA_BACKBONE_ARCH, config.ADA_BACKBONE_WEIGHTS_PATH,
            config.EMBEDDING_DIMENSION, config.DEVICE )
        self.faiss_handler = FaissDBHandler(config.EMBEDDING_DIMENSION)
        self.stats = defaultdict(int)

    def _get_image_paths_for_class(self, class_id_str: str) -> List[Path]: # Unchanged
        class_path = self.config.BASE_IMAGE_SOURCE_DIR / class_id_str
        img_paths: List[Path] = [];
        if not class_path.is_dir(): logger.warning(f"Class folder {class_path} not found."); return img_paths
        supported_ext = ['.png','.jpg','.jpeg','.webp','.bmp','.tiff']
        for item_path in class_path.rglob('*'):
            if item_path.is_file() and item_path.suffix.lower() in supported_ext:
                try:
                    with Image.open(item_path) as img_chk:
                        if img_chk.width<self.config.MIN_IMAGE_SIZE_TO_CONSIDER or img_chk.height<self.config.MIN_IMAGE_SIZE_TO_CONSIDER:
                            logger.debug(f"Orig img '{item_path.name}'({img_chk.size}) too small,skip.");self.stats[f'img_skip_small_orig_{class_id_str}']+=1;continue
                    img_paths.append(item_path)
                except UnidentifiedImageError:logger.warning(f"Corrupt img '{item_path.name}',skip.");self.stats[f'img_skip_corrupt_{class_id_str}']+=1
                except Exception as e:logger.error(f"Err open img '{item_path}':{e}")
        logger.info(f"Found {len(img_paths)} candidate images for class '{class_id_str}'.")
        return img_paths

    def build_prototype_faiss_index(self):
        logger.info(f"--- Building AdaFace FAISS (Prototypes) from '{self.config.BASE_IMAGE_SOURCE_DIR}' using MTCNN detector ---")
        self.stats.clear(); all_proto_embs:List[np.ndarray]=[]; all_proto_ids:List[str]=[]
        class_folders = [d for d in self.config.BASE_IMAGE_SOURCE_DIR.iterdir() if d.is_dir()]

        if not class_folders:
            logger.warning(f"No class subfolders in '{self.config.BASE_IMAGE_SOURCE_DIR}'. Empty FAISS DB will be created.")
            self.faiss_handler.create_and_save_index(np.array([]).reshape(0,self.config.EMBEDDING_DIMENSION),[],self.config.FAISS_INDEX_PATH,self.config.FAISS_MAP_PATH)
            return

        logger.info(f"Processing {len(class_folders)} class folders...")
        for class_dir in class_folders:
            class_id=class_dir.name; logger.info(f"Processing class: '{class_id}'"); self.stats['classes_scanned']+=1
            img_paths=self._get_image_paths_for_class(class_id)
            if not img_paths: logger.warning(f"No suitable images for '{class_id}', skipping."); self.stats['classes_skipped_no_imgs']+=1; continue
            
            class_embs:List[np.ndarray]=[]
            for img_path in img_paths:
                try:
                    img_pil_rgb = Image.open(img_path).convert("RGB") # MTCNN expects PIL image

                    # MODIFIED: Pass img_pil_rgb directly to MTCNN based detector
                    face_info = self.face_detector.detect_largest_face(img_pil_rgb)
                    
                    if not face_info:
                        logger.debug(f"No face (MTCNN) in '{img_path.name}' for '{class_id}'."); self.stats[f'img_skip_no_face_{class_id}']+=1; continue
                    
                    landmarks = face_info["landmarks"] # MTCNN from facenet-pytorch provides them in correct order for our alignment
                    img_np_rgb = np.array(img_pil_rgb) # Need NumPy for alignment
                    aligned_rgb_np = align_and_crop_face(img_np_rgb, landmarks)
                    if aligned_rgb_np is None:
                        logger.debug(f"Align fail for '{img_path.name}' in '{class_id}'."); self.stats[f'img_skip_align_fail_{class_id}']+=1; continue
                    
                    emb = self.embedding_extractor.extract_embedding(aligned_rgb_np)
                    if emb is not None: class_embs.append(emb); self.stats[f'embs_extracted_for_{class_id}']+=1
                    else: logger.debug(f"Embed fail for aligned '{img_path.name}'."); self.stats[f'img_skip_embed_fail_{class_id}']+=1
                except Exception as e: logger.error(f"Err proc img '{img_path.name}' for '{class_id}':{e}",exc_info=True); self.stats[f'img_skip_proc_err_{class_id}']+=1
            
            if class_embs:
                proto_emb=np.mean(np.array(class_embs),axis=0); norm_proto=normalize_vector(proto_emb)
                all_proto_embs.append(norm_proto); all_proto_ids.append(class_id)
                logger.info(f"Created proto for '{class_id}' from {len(class_embs)} embs."); self.stats['prototypes_created']+=1
            else: logger.warning(f"No embs for '{class_id}', proto not created."); self.stats['classes_fail_proto']+=1
        
        if not all_proto_embs:
            logger.error("No protos generated. FAISS index will be empty/not created.")
            self.faiss_handler.create_and_save_index(np.array([]).reshape(0,self.config.EMBEDDING_DIMENSION),[],self.config.FAISS_INDEX_PATH,self.config.FAISS_MAP_PATH)
        else:
            embs_np=np.array(all_proto_embs).astype(np.float32); ts=datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            idx_p,map_p=self.config.FAISS_INDEX_PATH,self.config.FAISS_MAP_PATH
            if idx_p.exists():shutil.move(str(idx_p),str(idx_p.parent/f"{idx_p.stem}_backup_{ts}{idx_p.suffix}"))
            if map_p.exists():shutil.move(str(map_p),str(map_p.parent/f"{map_p.stem}_backup_{ts}{map_p.suffix}"))
            self.faiss_handler.create_and_save_index(embs_np,all_proto_ids,idx_p,map_p)
            self.stats['total_prototypes_in_faiss']=embs_np.shape[0]

        logger.info("--- AdaFace FAISS Index Building (Prototypes, MTCNN) Completed ---")
        logger.info("Build Statistics:"); [logger.info(f"  {k}: {v}") for k,v in sorted(self.stats.items())]

if __name__ == "__main__":
    try:
        cfg = AdaFaceFaissBuilderConfig()
        logger.info(f"AdaFace FAISS Builder (MTCNN detector) starting. Device: {cfg.DEVICE}")
        builder = AdaFaceFaissBuilder(cfg)
        builder.build_prototype_faiss_index()
    except FileNotFoundError as e: logger.critical(f"Config Error: {e}. Check paths in Config."); sys.exit(1)
    except RuntimeError as e: logger.critical(f"Runtime Init Error: {e}. Check models, 'net.py'."); sys.exit(1)
    except Exception as e: logger.critical(f"Unexpected critical error: {e}", exc_info=True); sys.exit(1)
    logger.info("FAISS Builder script finished.")