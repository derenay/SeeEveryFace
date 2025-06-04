import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Union

import cv2
import faiss
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

# --- Face Detector: Using MTCNN from facenet-pytorch ---
try:
    from facenet_pytorch import MTCNN
    logging.info("Successfully imported MTCNN from facenet-pytorch.")
except ImportError:
    logging.critical("CRITICAL: 'facenet-pytorch' library not found. Please install it: pip install facenet-pytorch")
    raise
# --- End Face Detector Import ---

# --- CRITICAL: Import 'net' from AdaFace GitHub Repository ---
try:
    import net # This should import the net.py from the AdaFace repository
    logging.info(f"Successfully imported 'net.py' for AdaFace model definitions.")
except ImportError as e:
    logging.critical(f"CRITICAL IMPORT ERROR: Could not import 'net.py' from the AdaFace repository: {e}. "
                     "Ensure 'net.py' (and its dependencies) "
                     "from 'github.com/mk-minchul/AdaFace/' are in your Python path. ")
    raise
# --- End Critical Import Section ---

import queue
import threading
import uuid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - %(message)s')

class AppConfig:
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu" # For PyTorch models

    # Face Detector: MTCNN (facenet-pytorch)
    FACE_DETECTOR_MIN_FACE_SIZE: int = 10  # Minimum face size for MTCNN to detect
    FACE_DETECTOR_CONFIDENCE_THRESHOLD: float = 0.90 # MTCNN returns probabilities; filter on this

    # AdaFace Backbone Embedding Extractor
    ADA_BACKBONE_ARCH: str = 'ir_101' 
    ADA_BACKBONE_WEIGHTS_PATH: str = 'project-02/yenidenemem_few_shot/face_teslim/SOTA/models/adaface_ir101_webface12m.ckpt'
    EMBEDDING_DIMENSION: int = 512

    # FAISS Database
    FAISS_INDEX_FILE: str = 'project-02/yenidenemem_few_shot/face_teslim/SOTA/faissdb_mtcnn_ada/adaface_prototype_ir100_mtcnn.index'
    FAISS_MAP_FILE: str = 'project-02/yenidenemem_few_shot/face_teslim/SOTA/faissdb_mtcnn_ada/adaface_prototype_ir100_mtcnn_map.json'

    DEFAULT_RECOGNITION_THRESHOLD: float = 0.60 # Cosine DISTANCE (1.0 - cosine_similarity). Tune!

    VIDEO_PROCESS_EVERY_N_FRAMES: int = 3
    RESULTS_BASE_DIR: Path = Path("results_mtcnn_adaface_pipeline")
    SAVED_ALIGNED_FACES_DIR: Path = Path("run_detected_aligned_faces_mtcnn")

    FONT_PATH_WINDOWS: str = "arial.ttf"
    FONT_PATH_LINUX: str = "DejaVuSans.ttf"
    DEFAULT_FONT_SIZE: int = 15

    # Order: Left Eye, Right Eye, Nose, Mouth Left, Mouth Right
    ALIGNMENT_TARGET_LANDMARKS = np.array([
        [38.2946, 51.6963], [73.5318, 51.5014],
        [56.0252, 71.7366], [41.5493, 92.3655],
        [70.7299, 92.2041]], dtype=np.float32)
    ALIGNMENT_CROP_SIZE: Tuple[int, int] = (112, 112)

    @staticmethod
    def get_font_path() -> Optional[str]: # Unchanged
        font_path = AppConfig.FONT_PATH_WINDOWS if os.name == 'nt' else AppConfig.FONT_PATH_LINUX
        try: ImageFont.truetype(font_path, AppConfig.DEFAULT_FONT_SIZE); return font_path
        except IOError: logging.warning(f"Font '{font_path}' not found."); return None

def normalize_vector(vec: np.ndarray) -> np.ndarray: # Unchanged
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-6 else np.zeros_like(vec).astype('float32')

def load_pil_font(size: int = 15) -> ImageFont.FreeTypeFont: # Unchanged
    font_path = AppConfig.get_font_path()
    if font_path:
        try: return ImageFont.truetype(font_path, size)
        except IOError: pass
    try: return ImageFont.load_default(size=size)
    except TypeError: return ImageFont.load_default()
    except Exception as e: raise RuntimeError(f"Font loading failed: {e}") from e

def align_and_crop_face(image_np_rgb: np.ndarray, landmarks_5pts_ordered: np.ndarray,
                        target_landmarks: np.ndarray = AppConfig.ALIGNMENT_TARGET_LANDMARKS,
                        crop_size: Tuple[int, int] = AppConfig.ALIGNMENT_CROP_SIZE) -> Optional[np.ndarray]:
    # Unchanged
    try:
        transform_matrix, _ = cv2.estimateAffinePartial2D(landmarks_5pts_ordered.astype(np.float32), target_landmarks.astype(np.float32))
        if transform_matrix is None: return None
        aligned_face_rgb = cv2.warpAffine(image_np_rgb, transform_matrix, crop_size, borderValue=0.0)
        return aligned_face_rgb
    except cv2.error as e: logging.debug(f"OpenCV align error: {e}"); return None
    except Exception as e: logging.error(f"Align error: {e}", exc_info=False); return None

class ImageSaver: # Unchanged
    def __init__(self, base_save_dir: Path):
        self.save_dir=base_save_dir; self.save_dir.mkdir(parents=True,exist_ok=True)
        self.image_queue:queue.Queue=queue.Queue(maxsize=100);self.stop_event=threading.Event()
        self.worker_thread=threading.Thread(target=self._worker,daemon=True);self.worker_thread.start()
        logging.info(f"ImageSaver (aligned faces) active. Dir: {self.save_dir}")
    def _generate_filename(self,fn:Optional[int]=None,di:Optional[int]=None)->str:
        ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f");parts=["aligned_face"]
        if fn is not None:parts.append(f"f{fn}")
        if di is not None:parts.append(f"d{di}")
        parts.append(ts);parts.append(str(uuid.uuid4().hex)[:6]);return "_".join(parts)+".png"
    def _worker(self):
        while not self.stop_event.is_set() or not self.image_queue.empty():
            try:
                img,fn,di=self.image_queue.get(timeout=0.1);fp=self.save_dir/self._generate_filename(fn,di)
                try:img.save(fp,"PNG")
                except Exception as e:logging.error(f"Save fail {fp}:{e}")
                finally:self.image_queue.task_done()
            except queue.Empty:
                if self.stop_event.is_set() and self.image_queue.empty():break
            except Exception as e:logging.error(f"ImageSaver worker error:{e}")
    def schedule_save(self,img_pil:Image.Image,fn:Optional[int]=None,di:Optional[int]=None):
        if self.stop_event.is_set():return
        try:self.image_queue.put_nowait((img_pil,fn,di))
        except queue.Full:logging.warning(f"ImageSaver Q full. Face(f:{fn},d:{di}) skip.")
    def stop(self,wait:bool=True):
        logging.info("ImageSaver stopping...");self.stop_event.set()
        if wait:self.image_queue.join()
        self.worker_thread.join(timeout=3.0)
        if self.worker_thread.is_alive():logging.warning("ImageSaver worker timed out.")
        logging.info("ImageSaver stopped.")

class FaceDetector: # MODIFIED to use MTCNN from facenet-pytorch
    def __init__(self, device_str: str, min_face_size: int, confidence_threshold: float):
        self.device = torch.device(device_str)
        self.min_face_size = min_face_size
        self.confidence_threshold = confidence_threshold # Applied to MTCNN's output probability
        self.mtcnn: Optional[MTCNN] = None
        self._load_model()

    def _load_model(self):
        try:
            self.mtcnn = MTCNN(
                keep_all=True,       # Detect all faces in the image
                device=self.device,
                min_face_size=self.min_face_size,
                post_process=False,  # Get raw landmark coordinates, not normalized tensors
                select_largest=False # We process all faces that meet confidence
            )
            logging.info(f"MTCNN detector (facenet-pytorch) initialized on {self.device}.")
        except Exception as e:
            logging.error(f"Failed to initialize MTCNN: {e}", exc_info=True)
            raise RuntimeError(f"MTCNN model initialization failed: {e}")

    def detect_faces(self, image_pil_rgb: Image.Image) -> List[Dict[str, Any]]:
        """
        Detects faces using MTCNN.
        Input: image_pil_rgb (PIL Image in RGB order).
        Output: List of dicts with 'box', 'landmarks' (ordered for alignment), 'confidence'.
        """
        detected_faces_info: List[Dict[str, Any]] = []
        if self.mtcnn is None:
            logging.error("MTCNN model not loaded.")
            return detected_faces_info
        if image_pil_rgb is None:
            logging.warning("FaceDetector (MTCNN) received None image.")
            return detected_faces_info
            
        try:
            # MTCNN's detect method returns: boxes (Nx4), probs (N,), landmarks (Nx5x2)
            # It expects a PIL image.
            boxes, probs, landmarks_all = self.mtcnn.detect(image_pil_rgb, landmarks=True)
            
            if boxes is not None: # If any faces are detected
                for i in range(len(boxes)):
                    if probs[i] is None or probs[i] < self.confidence_threshold:
                        continue

                    box = boxes[i]  # [x1, y1, x2, y2]
                    x1, y1, x2, y2 = map(int, box)

                    if x2 > x1 and y2 > y1: # Valid box
                        # Landmarks from facenet-pytorch MTCNN are [left_eye, right_eye, nose, mouth_left, mouth_right]
                        # This order matches AppConfig.ALIGNMENT_TARGET_LANDMARKS.
                        landmarks_np = landmarks_all[i].astype(np.int32) # Already 5x2 (x,y)
                        
                        detected_faces_info.append({
                            "box": [x1, y1, x2, y2],
                            "landmarks": landmarks_np, # Correctly ordered
                            "confidence": float(probs[i])
                        })
        except Exception as e:
            logging.error(f"Error during MTCNN face detection: {e}", exc_info=True)
        return detected_faces_info

class EmbeddingExtractor: # Uses AdaFace backbone from .ckpt via net.py (Unchanged)
    def __init__(self, model_arch: str, weights_path: str, embedding_dim: int, device: str):
        self.model_arch_name=model_arch; self.weights_path=Path(weights_path)
        self.embedding_dim=embedding_dim; self.device=torch.device(device)
        self.model:Optional[torch.nn.Module]=None; self._load_model()
    def _load_model(self):
        if not self.weights_path.exists(): logging.critical(f"AdaFace .ckpt NOT FOUND:{self.weights_path}");raise FileNotFoundError(f"AdaFace .ckpt missing:{self.weights_path}")
        try:
            logging.info(f"Loading AdaFace arch:'{self.model_arch_name}' via net.build_model.")
            self.model=net.build_model(self.model_arch_name)
            ckpt=torch.load(self.weights_path,map_location=torch.device('cpu'))
            s_dict=ckpt.get('state_dict');
            if not s_dict:raise KeyError("'state_dict' missing in .ckpt.")
            model_s_dict={k[6:]:v for k,v in s_dict.items() if k.startswith('model.')}
            if not model_s_dict and s_dict:model_s_dict=s_dict
            if not model_s_dict:raise ValueError("Could not prep valid state_dict from .ckpt.")
            self.model.load_state_dict(model_s_dict,strict=True)
            self.model.to(self.device);self.model.eval()
            logging.info(f"AdaFace backbone '{self.model_arch_name}' loaded from '{self.weights_path}'.")
            dummy_rgb_np=np.zeros((AppConfig.ALIGNMENT_CROP_SIZE[0],AppConfig.ALIGNMENT_CROP_SIZE[1],3),dtype=np.uint8)
            dummy_input=self._preprocess_input(dummy_rgb_np)
            with torch.no_grad():_,_=self.model(dummy_input.to(self.device))
            logging.info("AdaFace backbone warmed up.")
        except Exception as e: logging.critical(f"CRITICAL Error loading AdaFace backbone:{e}",exc_info=True);raise RuntimeError(f"Failed to load AdaFace backbone:{e}")
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
        except Exception as e: logging.warning(f"AdaFace embedding extraction error:{e}",exc_info=False);return None

class FaissDatabase: # Unchanged
    def __init__(self,index_path:str,map_path:str,expected_embedding_dim:int):
        self.index_path=Path(index_path);self.map_path=Path(map_path)
        self.expected_embedding_dim=expected_embedding_dim
        self.faiss_index:Optional[faiss.Index]=None;self.index_to_name_map:List[str]=[]
        if self.expected_embedding_dim<=0:raise ValueError("Invalid emb_dim.")
        self._load_database()
    def _load_database(self):
        self.index_path.parent.mkdir(parents=True,exist_ok=True);self.map_path.parent.mkdir(parents=True,exist_ok=True)
        logging.info(f"Loading FAISS. Idx:'{self.index_path}',Map:'{self.map_path}'")
        if self.map_path.exists():
            try:
                with open(self.map_path,'r',encoding='utf-8')as f:self.index_to_name_map=json.load(f)
                logging.info(f"FAISS map loaded({len(self.index_to_name_map)} entries).")
            except Exception as e:raise RuntimeError(f"Load FAISS map fail:{e}")
        else:logging.warning(f"FAISS map {self.map_path} not found.")
        if self.index_path.exists():
            try:
                self.faiss_index=faiss.read_index(str(self.index_path))
                logging.info(f"FAISS idx loaded({self.faiss_index.ntotal} vec,dim:{self.faiss_index.d}).")
                if self.faiss_index.ntotal>0 and self.faiss_index.ntotal!=len(self.index_to_name_map):
                    logging.warning(f"FAISS idx/map size mismatch!({self.faiss_index.ntotal} vs {len(self.index_to_name_map)})")
                if self.faiss_index.d!=self.expected_embedding_dim:
                    raise ValueError(f"FAISS idx dim({self.faiss_index.d})!=expected({self.expected_embedding_dim})!")
            except Exception as e:raise RuntimeError(f"Load FAISS idx fail:{e}")
        else:logging.warning(f"FAISS idx {self.index_path} not found.Rec disabled.")
        logging.info("FAISS DB init done.")
    def search(self,query_emb:np.ndarray,k:int=1)->Tuple[Optional[np.ndarray],Optional[np.ndarray]]:
        if not self.faiss_index or self.faiss_index.ntotal==0:return None,None
        try:return self.faiss_index.search(np.array([query_emb],dtype=np.float32),k)
        except Exception as e:logging.error(f"FAISS search err:{e}");return None,None
    def get_identity(self,idx_db:int)->Optional[str]:
        return self.index_to_name_map[idx_db] if 0<=idx_db<len(self.index_to_name_map) else None

class FaceRecognitionService: # Input to detector is PIL RGB for MTCNN
    def __init__(self,face_detector:FaceDetector,extractor:EmbeddingExtractor,database:FaissDatabase,rec_thresh:float,img_saver:Optional[ImageSaver]=None):
        self.face_detector=face_detector;self.extractor=extractor;self.database=database
        self.recognition_threshold_distance=rec_thresh;self.image_saver=img_saver
        self.current_frame_number_for_saving:Optional[int]=None
    def set_current_frame_number(self,fn:int):self.current_frame_number_for_saving=fn
    def _draw_annotations(self,draw:ImageDraw.Draw,fh:int,rd:Dict[str,Any]): # Unchanged
        x1,y1,x2,y2=rd["box_absolute"];idt=rd["identity"];st=rd["status"];sim=rd.get("similarity")
        clr="lime" if st=="Recognized" else ("orange" if st=="Unknown" else ("yellow" if st=="NoEmbedding" else "red"))
        lbl=f"{idt}" + (f" (S:{sim:.2f})" if sim is not None and st!="NoEmbedding" else "")
        fs=max(12,int(fh/45));font=load_pil_font(size=fs);draw.rectangle([(x1,y1),(x2,y2)],outline=clr,width=max(1,int(fh/300)+1))
        try:tb=draw.textbbox((x1,y1-fs-3),lbl,font=font)
        except AttributeError:tl=draw.textlength(lbl,font=font) if hasattr(draw,'textlength') else len(lbl)*fs*0.6;th=fs*1.2;tb=(x1,y1-fs-3,x1+int(tl),y1-3+int(th))
        tw=tb[2]-tb[0];th=tb[3]-tb[1];bgy1=y1-th-4;bgy2=y1-2
        if bgy1<0:bgy1=y2+2;bgy2=y2+th+4
        draw.rectangle([(x1,bgy1),(x1+tw+4,bgy2)],fill=clr);draw.text((x1+2,bgy1),lbl,fill="black",font=font)

    def recognize_faces_in_frame(self, full_frame_pil_rgb: Image.Image) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        annotated_frame_pil = full_frame_pil_rgb.copy()
        draw = ImageDraw.Draw(annotated_frame_pil)
        recognition_results: List[Dict[str, Any]] = []

        # MTCNN detector expects PIL Image
        detected_faces = self.face_detector.detect_faces(full_frame_pil_rgb)

        full_frame_np_rgb = np.array(full_frame_pil_rgb) # For alignment, if faces are detected

        for idx, face_info in enumerate(detected_faces):
            abs_box = face_info["box"]
            # Landmarks from MTCNN (facenet-pytorch) are already in the correct order for our alignment
            ordered_landmarks = face_info["landmarks"]
            
            aligned_face_np_rgb = align_and_crop_face(full_frame_np_rgb, ordered_landmarks)
            if aligned_face_np_rgb is None: continue

            if self.image_saver:
                self.image_saver.schedule_save(Image.fromarray(aligned_face_np_rgb),
                                               self.current_frame_number_for_saving, idx)
            
            query_embedding = self.extractor.extract_embedding(aligned_face_np_rgb)
            identity,status,sim_score,closest_match="Bilinmeyen","NoEmbedding",None,None
            if query_embedding is not None:
                status="Unknown"
                if self.database.faiss_index and self.database.faiss_index.ntotal>0:
                    sims,inds=self.database.search(query_embedding,k=1)
                    if inds is not None and sims is not None and inds.size>0:
                        db_idx,sim_score=inds[0][0],float(sims[0][0]);sim_score=max(0.0,min(1.0,sim_score))
                        ret_name=self.database.get_identity(db_idx)
                        if ret_name:
                            closest_match=ret_name
                            if (1.0-sim_score)<self.recognition_threshold_distance:identity,status=ret_name,"Recognized"
                else:status="NoIndex"
            
            result={"box_absolute":abs_box,"identity":identity,"status":status,"similarity":sim_score,
                      "closest_match_debug":closest_match,"detection_confidence":face_info.get("confidence")}
            recognition_results.append(result)
            self._draw_annotations(draw,full_frame_pil_rgb.height,result)
            
        return annotated_frame_pil, recognition_results

class MediaProcessor: # Unchanged
    def __init__(self,rs:FaceRecognitionService,obd:Path):self.recognition_service=rs;self.output_base_dir=obd;self.output_base_dir.mkdir(parents=True,exist_ok=True)
    def _create_ts_dir(self,pfx="session")->Path:d=self.output_base_dir/f"{pfx}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}";d.mkdir(parents=True,exist_ok=True);logging.info(f"Out dir:{d}");return d
    def _save_json(self,od:Path,data:Dict,npfx="results"):
        p=od/f"{npfx}.json";
        try:
            def cvt_path(o):return str(o.resolve()) if isinstance(o,Path) else f"Unserializable_{type(o).__name__}"
            with open(p,'w',encoding='utf-8')as f:json.dump(data,f,indent=4,ensure_ascii=False,default=cvt_path)
            logging.info(f"JSON saved:{p}")
        except Exception as e:logging.error(f"JSON save fail {p}:{e}")
    def process_image(self,imp_str:str):
        imp=Path(imp_str);logging.info(f"Proc img:{imp}")
        if not imp.exists():logging.error(f"Img not found:{imp}");return
        od=self._create_ts_dir(f"img_{imp.stem}")
        try:img_pil=Image.open(imp).convert("RGB")
        except Exception as e:logging.error(f"Open img err {imp}:{e}");return
        if self.recognition_service.image_saver:self.recognition_service.set_current_frame_number(0)
        st=time.time();ann_img,res=self.recognition_service.recognize_faces_in_frame(img_pil);pt=round(time.time()-st,3)
        logging.info(f"Img proc time:{pt:.3f}s.")
        try:ann_img.save(od/f"annotated_{imp.name}")
        except Exception as e:logging.error(f"Save ann img err:{e}")
        jd={"src":imp,"time_s":pt,"thresh_dist":self.recognition_service.recognition_threshold_distance,
            "models":{"detector":"MTCNN (facenet-pytorch)","embedder":AppConfig.ADA_BACKBONE_ARCH},"recognitions":res}
        self._save_json(od,jd)
    def process_video(self,vp_str:str,nfs:int):
        vp=Path(vp_str);logging.info(f"Proc vid:{vp}(every {nfs}f)")
        if not vp.exists():logging.error(f"Vid not found:{vp}");return
        od=self._create_ts_dir(f"vid_{vp.stem}");cap=cv2.VideoCapture(str(vp))
        if not cap.isOpened():logging.error(f"Vid open err:{vp}");return
        fps=cap.get(cv2.CAP_PROP_FPS)or 30.0;w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tf=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));ovp=od/f"ann_{vp.stem}.mp4"
        wr=cv2.VideoWriter(str(ovp),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h)) if w>0 and h>0 else None
        fi,pc,ar,st=0,0,[],time.time()
        while True:
            ret,f_bgr=cap.read();
            if not ret:break
            fi+=1;prg="-\\|/"[fi%4];print(f"\r{prg}Frame:{fi}/{tf if tf>0 else'?'}(An:{pc})",end="")
            out_f=f_bgr
            if fi%nfs==0:
                pc+=1
                if self.recognition_service.image_saver:self.recognition_service.set_current_frame_number(fi)
                img_p=Image.fromarray(cv2.cvtColor(f_bgr,cv2.COLOR_BGR2RGB)) # MTCNN gets PIL RGB
                ann_p,f_res=self.recognition_service.recognize_faces_in_frame(img_p);out_f=cv2.cvtColor(np.array(ann_p),cv2.COLOR_RGB2BGR)
                if f_res:ar.append({"frame":fi,"recognitions":f_res})
            if wr:wr.write(out_f)
        print("\nVid done.");pt=round(time.time()-st,2);cap.release()
        if wr:wr.release();logging.info(f"Proc vid saved:{ovp}")
        jd={"src":vp,"time_s":pt,"total_f":fi,"analyzed_f":pc,
            "models":{"detector":"MTCNN (facenet-pytorch)","embedder":AppConfig.ADA_BACKBONE_ARCH},"timeline":ar}
        self._save_json(od,jd,f"vid_{vp.stem}")
    def process_live_stream(self,src:Union[int,str],nfs:int): # Unchanged
        logging.info(f"Live stream:{src}(every {nfs}f)")
        cap=cv2.VideoCapture(src)
        if not cap.isOpened():logging.error(f"Live stream err:{src}");return
        win="Live Face Rec (MTCNN+AdaFace)";cv2.namedWindow(win,cv2.WINDOW_NORMAL);fi=0;lft=time.time();ffp=0
        while True:
            ret,f_bgr=cap.read()
            if not ret:logging.warning("Stream end/err.");time.sleep(1);continue
            fi+=1;ffp+=1;out_f=f_bgr.copy()
            if fi%nfs==0:
                if self.recognition_service.image_saver:self.recognition_service.set_current_frame_number(fi)
                img_p=Image.fromarray(cv2.cvtColor(f_bgr,cv2.COLOR_BGR2RGB)) # MTCNN gets PIL RGB
                ann_p,_=self.recognition_service.recognize_faces_in_frame(img_p);out_f=cv2.cvtColor(np.array(ann_p),cv2.COLOR_RGB2BGR)
            ct=time.time()
            if(ct-lft)>=1.0:dfps=ffp/(ct-lft);lft=ct;ffp=0;cv2.putText(out_f,f"FPS:{dfps:.1f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            cv2.imshow(win,out_f)
            if cv2.waitKey(1)&0xFF==ord('q'):logging.info("User quit.");break
        cap.release();cv2.destroyAllWindows();logging.info("Live stream finished.")

def main_application():
    config = AppConfig()
    if "PLEASE_UPDATE_THIS_PATH" in str(config.ADA_BACKBONE_WEIGHTS_PATH) or \
       not Path(config.ADA_BACKBONE_WEIGHTS_PATH).is_file():
        msg=f"CRITICAL:AppConfig.ADA_BACKBONE_WEIGHTS_PATH ('{config.ADA_BACKBONE_WEIGHTS_PATH}') err.Update path."
        logging.critical(msg);print(msg);return
    logging.info(f"AdaFace .ckpt:{config.ADA_BACKBONE_WEIGHTS_PATH}")
    logging.info(f"AdaFace arch:'{config.ADA_BACKBONE_ARCH}' (Ensure 'net.py' is imported)")
    Path(config.FAISS_INDEX_FILE).parent.mkdir(parents=True,exist_ok=True)
    if config.SAVED_ALIGNED_FACES_DIR:config.SAVED_ALIGNED_FACES_DIR.mkdir(parents=True,exist_ok=True)

    img_saver:Optional[ImageSaver]=None
    if config.SAVED_ALIGNED_FACES_DIR:img_saver=ImageSaver(config.SAVED_ALIGNED_FACES_DIR)

    try:
        logging.info(f"Init components on device:{config.DEVICE}")
        face_detector = FaceDetector( # MTCNN instantiation
            device_str=config.DEVICE, # MTCNN can use 'cuda' or 'cpu'
            min_face_size=config.FACE_DETECTOR_MIN_FACE_SIZE,
            confidence_threshold=config.FACE_DETECTOR_CONFIDENCE_THRESHOLD
        )
        embed_extractor=EmbeddingExtractor(
            model_arch=config.ADA_BACKBONE_ARCH,weights_path=config.ADA_BACKBONE_WEIGHTS_PATH,
            embedding_dim=config.EMBEDDING_DIMENSION,device=config.DEVICE)
        faiss_db=FaissDatabase(
            index_path=config.FAISS_INDEX_FILE,map_path=config.FAISS_MAP_FILE,
            expected_embedding_dim=config.EMBEDDING_DIMENSION)
        rec_service=FaceRecognitionService(
            face_detector,embed_extractor,faiss_db,
            config.DEFAULT_RECOGNITION_THRESHOLD,img_saver)
        media_proc=MediaProcessor(rec_service,config.RESULTS_BASE_DIR)
        logging.info("All components initialized.")

        input_media:Union[str,int]="project-02/yenidenemem_few_shot/face_teslim/SOTA/test_video_images/vesikalık.png" 
        # input_media:Union[str,int]="path/to/your/image.jpg"
        # input_media:Union[str,int]="path/to/your/video.mp4"
        
        if isinstance(input_media,str) and "path/to/your/" in input_media:
            logging.error(f"Input path '{input_media}' placeholder.Update.");return
        if isinstance(input_media,str) and not input_media.lower().startswith(("rtsp://","http://","https://")) and not Path(input_media).exists():
            logging.error(f"Input file not found:'{input_media}'.");return

        is_live=isinstance(input_media,int) or \
                  (isinstance(input_media,str) and input_media.lower().startswith(("rtsp://","http://","https://","/dev/video")))
        
        if is_live:media_proc.process_live_stream(input_media,config.VIDEO_PROCESS_EVERY_N_FRAMES)
        elif isinstance(input_media,str) and Path(input_media).is_file():
            ext=Path(input_media).suffix.lower()
            if ext in ['.jpg','.jpeg','.png','.bmp','.webp']:media_proc.process_image(input_media)
            elif ext in ['.mp4','.avi','.mov','.mkv']:media_proc.process_video(input_media,config.VIDEO_PROCESS_EVERY_N_FRAMES)
            else:logging.error(f"Unsupported file:{input_media}")
        else:logging.error(f"Invalid input media:{input_media}")

    except Exception as e:logging.critical(f"App error:{e}",exc_info=True);print(f"CRITICAL ERROR:{e}.")
    finally:
        if img_saver:img_saver.stop()
        logging.info("Application shutdown.")

if __name__=='__main__':
    main_application()