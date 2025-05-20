import datetime
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import cv2
# import faiss # FAISS artık kullanılmayacak
import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from ultralytics import YOLO
from deepface import DeepFace
import queue
import threading
import uuid
import joblib # MLP modelini yüklemek için
from sklearn.preprocessing import LabelEncoder # LabelEncoder'ı yüklemek için
from sklearn.neural_network import MLPClassifier # MLPClassifier'ın tanınması için (joblib yüklerken gerekebilir)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)

class AppConfig:
    EMBEDDING_MODEL_NAME: str = 'Facenet512'
    YOLO_FACE_MODEL_PATH: str = 'project-02/yolov11m-face.pt'
    DEEPFACE_DETECTOR_BACKEND_FOR_CROPS: str = 'skip'
    
    # FAISS yerine MLP için eşik ve yollar
    MLP_CONFIDENCE_THRESHOLD: float = 0.60 # MLP için minimum güven skoru (örn: %60)
    
    HUMANS_BASE_DIR: Path = Path("project-02/yenidenemem_few_shot/face_teslim/face_db/Human/") # Ana insan veritabanı yolu
    SAVED_MODELS_SUBDIR_NAME = "saved_models" 
    MLP_MODEL_FILENAME = "mlp_classifier.joblib"
    MLP_ENCODER_FILENAME = "label_encoder.joblib"

    RESULTS_BASE_DIR: Path = Path("results_face_professional_mlp") # Çıktı klasörü adı değişebilir
    YOLO_CONFIDENCE_THRESHOLD: float = 0.5
    MIN_FACE_SIZE_FOR_EMBEDDING: int = 30 
    VIDEO_PROCESS_EVERY_N_FRAMES: int = 4 
    DETECTED_FACES_SUBDIR: str = "detected" # Bu hala ImageSaver için kullanılabilir
    
    FONT_PATH_WINDOWS: str = "arial.ttf"
    FONT_PATH_LINUX: str = "DejaVuSans.ttf"
    DEFAULT_FONT_SIZE: int = 15

    def __init__(self):
        # MLP Model ve Encoder yollarını dinamik olarak oluştur
        self.SAVED_MODELS_DIR = self.HUMANS_BASE_DIR / self.SAVED_MODELS_SUBDIR_NAME
        self.MLP_MODEL_PATH = self.SAVED_MODELS_DIR / self.MLP_MODEL_FILENAME
        self.MLP_ENCODER_PATH = self.SAVED_MODELS_DIR / self.MLP_ENCODER_FILENAME
        
        # Diğer dinamik yollar
        self.DETECTED_FACES_SAVE_DIR = self.HUMANS_BASE_DIR / self.DETECTED_FACES_SUBDIR


    @staticmethod
    def get_font_path() -> Optional[str]:
        font_path = AppConfig.FONT_PATH_WINDOWS if os.name == 'nt' else AppConfig.FONT_PATH_LINUX
        try: ImageFont.truetype(font_path, AppConfig.DEFAULT_FONT_SIZE); return font_path
        except IOError: logger.warning(f"Font ('{font_path}') bulunamadı."); return None

# --- Yardımcı Fonksiyonlar ---
def normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-6: return np.zeros_like(vec).astype('float32')
    return (vec / norm).astype('float32')

def load_pil_font(size: int = 15) -> ImageFont.FreeTypeFont:
    font_path = AppConfig.get_font_path()
    if font_path:
        try: return ImageFont.truetype(font_path, size)
        except IOError: pass
    try: return ImageFont.load_default(size=size) # Pillow 10+
    except TypeError: return ImageFont.load_default() # Pillow < 10

class ImageSaver:
    # ... (ImageSaver kodu öncekiyle aynı - değişiklik yok) ...
    def __init__(self, base_save_dir: Path):
        self.save_dir = base_save_dir; self.save_dir.mkdir(parents=True, exist_ok=True)
        self.image_queue = queue.Queue(maxsize=100); self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True); self.worker_thread.start()
        logger.info(f"ImageSaver başlatıldı. Kayıt dizini: {self.save_dir}")
    def _generate_filename(self, frame_number: Optional[int]=None, detection_idx: Optional[int]=None) -> str:
        ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"); parts=["face"]
        if frame_number is not None: parts.append(f"f{frame_number}")
        if detection_idx is not None: parts.append(f"d{detection_idx}")
        parts.append(ts); return "_".join(parts)+".png"
    def _worker(self):
        while not self.stop_event.is_set() or not self.image_queue.empty():
            try:
                img_pil,frame_num,det_idx = self.image_queue.get(timeout=0.1)
                fp = self.save_dir/self._generate_filename(frame_num,det_idx)
                try: img_pil.save(fp); logger.debug(f"Yüz kaydedildi: {fp}")
                except Exception as e: logger.error(f"Yüz {fp} kaydedilemedi: {e}")
                finally: self.image_queue.task_done()
            except queue.Empty:
                if self.stop_event.is_set(): break
            except Exception as e: logger.error(f"ImageSaver worker hatası: {e}")
    def schedule_save(self, img_pil:Image.Image, frame_number:Optional[int]=None, detection_idx:Optional[int]=None):
        if self.stop_event.is_set(): logger.warning("ImageSaver durduruldu, kayıt planlanamaz."); return
        try: self.image_queue.put_nowait((img_pil,frame_number,detection_idx))
        except queue.Full: logger.warning(f"ImageSaver kuyruğu dolu. Yüz (kare:{frame_number}, idx:{detection_idx}) kaydedilemedi.")
    def stop(self, wait_for_completion:bool=True):
        logger.info("ImageSaver durduruluyor..."); self.stop_event.set()
        if wait_for_completion: self.image_queue.join()
        self.worker_thread.join(timeout=2.0)
        if self.worker_thread.is_alive(): logger.warning("ImageSaver worker zamanında durmadı.")
        logger.info("ImageSaver durduruldu.")

class FaceDetector:
    # ... (FaceDetector kodu öncekiyle aynı - değişiklik yok) ...
    def __init__(self, model_path: str, confidence_threshold: float):
        self.model_path = Path(model_path); self.confidence_threshold = confidence_threshold
        self.model: Optional[YOLO] = None; self._load_model()
    def _load_model(self):
        if not self.model_path.exists(): logger.error(f"YOLO modeli {self.model_path} bulunamadı."); raise FileNotFoundError(f"YOLO {self.model_path}")
        try: self.model = YOLO(self.model_path); logger.info(f"YOLO modeli '{self.model_path}' yüklendi.")
        except Exception as e: logger.error(f"YOLO yüklenemedi: {e}"); raise RuntimeError(f"YOLO yüklenemedi: {e}")
    def detect_faces(self, image_pil: Image.Image) -> List[Dict[str, Any]]:
        if not self.model: logger.error("YOLO modeli yüklenmemiş."); return []
        detected_faces = []
        try: results = self.model.predict(image_pil, verbose=False, conf=self.confidence_threshold)
        except Exception as e: logger.error(f"YOLO predict hatası: {e}"); return []
        if results:
            for res_set in results:
                if hasattr(res_set,'boxes') and res_set.boxes:
                    for box in res_set.boxes:
                        if box.xyxy is not None and len(box.xyxy)>0 and box.conf is not None and len(box.conf)>0:
                            x1,y1,x2,y2 = map(int,box.xyxy[0].tolist())
                            if x1<x2 and y1<y2: detected_faces.append({"box":[x1,y1,x2,y2],"confidence":float(box.conf[0])})
        return detected_faces

class EmbeddingExtractor: # MLP için olanla aynı, AppConfig'den MIN_FACE_SIZE_FOR_EMBEDDING kullanacak
    def __init__(self, model_name: str, detector_backend: str, min_face_size: int):
        self.model_name,self.detector_backend,self.min_face_size=model_name,detector_backend,min_face_size
        self.emb_dim:int=-1; self._init_deepface()
        logger.info(f"Extractor: {self.model_name} (Min:{self.min_face_size}px, Dim:{self.emb_dim})")
    def _init_deepface(self):
        try:
            res=DeepFace.represent(np.zeros((max(64,self.min_face_size),max(64,self.min_face_size),3),dtype=np.uint8), model_name=self.model_name, enforce_detection=False, detector_backend=self.detector_backend, align=False)
            self.emb_dim = len(res[0]["embedding"]) if isinstance(res,list) and res and "embedding" in res[0] else {'Facenet512':512, 'VGG-Face':2622}.get(self.model_name,512) # VGG-Face için 2622, Facenet512 için 512
        except Exception as e: logger.error(f"DeepFace modeli '{self.model_name}' yüklenemedi: {e}"); raise RuntimeError(f"DeepFace yüklenemedi: {e}")
    
    def get_embedding_dimension(self) -> int:
        if self.emb_dim <= 0: raise RuntimeError("Embedding boyutu belirlenemedi.")
        return self.emb_dim

    def extract_embedding(self, face_crop_pil: Image.Image) -> Optional[np.ndarray]: # PIL Image alır
        try:
            if face_crop_pil.width < self.min_face_size or face_crop_pil.height < self.min_face_size: 
                logger.debug(f"Yüz kırpıntısı çok küçük ({face_crop_pil.size}), embedding atlanıyor.")
                return None
            # DeepFace.represent zaten numpy array veya image path alabilir.
            # PIL Image'ı numpy array'e çevirelim (RGB olduğundan emin olalım)
            img_np_rgb = np.array(face_crop_pil.convert("RGB"))

            res = DeepFace.represent(img_np_rgb, model_name=self.model_name, enforce_detection=False, 
                                     detector_backend=self.detector_backend, align=False)
            if isinstance(res, list) and res and "embedding" in res[0] and len(res[0]["embedding"]) == self.emb_dim:
                return np.array(res[0]["embedding"])
            logger.warning(f"Geçerli embedding alınamadı. Dönen: {type(res)}")
            return None
        except UnidentifiedImageError: logger.error(f"Geçersiz görüntü dosyası (PIL).")
        except Exception as e:
            logger.error(f"Embedding çıkarılırken hata: {e}")
            return None

# --- YENİ: Eğitilmiş MLP Sınıflandırıcı ---
class TrainedMLPClassifier:
    def __init__(self, model_path: Path, encoder_path: Path):
        self.model: Optional[MLPClassifier] = None
        self.label_encoder: Optional[LabelEncoder] = None
        if model_path.exists() and encoder_path.exists():
            try:
                self.model = joblib.load(model_path)
                self.label_encoder = joblib.load(encoder_path)
                logger.info(f"MLP modeli '{model_path}' ve encoder '{encoder_path}' başarıyla yüklendi.")
                logger.info(f"Modelin tanıdığı sınıflar ({len(self.label_encoder.classes_)} adet): {list(self.label_encoder.classes_)}")
            except Exception as e:
                logger.error(f"Kaydedilmiş MLP model/encoder yüklenirken hata: {e}")
        else:
            logger.error(f"MLP Model ({model_path}) veya Label Encoder ({encoder_path}) bulunamadı. Lütfen önce modeli eğitin.")

    def is_ready(self) -> bool:
        return self.model is not None and self.label_encoder is not None

    def predict(self, embedding: np.ndarray) -> Optional[Tuple[str, float]]:
        if not self.is_ready() or not self.model or not self.label_encoder: # type hinting için
            logger.error("MLP Modeli veya Encoder yüklenmediği için tahmin yapılamıyor.")
            return None
        try:
            embedding_reshaped = embedding.reshape(1, -1)
            probabilities = self.model.predict_proba(embedding_reshaped)[0]
            predicted_encoded_label = np.argmax(probabilities)
            confidence = probabilities[predicted_encoded_label]
            predicted_original_label = self.label_encoder.inverse_transform([predicted_encoded_label])[0]
            return predicted_original_label, float(confidence)
        except Exception as e:
            logger.error(f"MLP ile tahmin yapılırken hata: {e}")
            return None

class FaceRecognitionService:
    def __init__(self, detector: FaceDetector, extractor: EmbeddingExtractor,
                 mlp_classifier: TrainedMLPClassifier, # FAISS yerine MLP
                 recognition_threshold: float, # Bu artık MLP güven eşiği olacak
                 image_saver: Optional[ImageSaver] = None):
        self.detector = detector
        self.extractor = extractor
        self.mlp_classifier = mlp_classifier # YENİ
        self.recognition_threshold = recognition_threshold # Artık MLP güven eşiği
        self.image_saver = image_saver 
        self.current_frame_number_for_saving: Optional[int] = None 

    def set_current_frame_number(self, frame_number: int):
        self.current_frame_number_for_saving = frame_number

    def _draw_annotations(self, draw: ImageDraw.Draw, frame_height: int, result_data: Dict[str, Any]):
        x1, y1, x2, y2 = result_data["box"]
        identity = result_data["identity"]
        status = result_data["status"]
        confidence = result_data.get("confidence") # Artık confidence kullanıyoruz
        
        color = "red"
        if status == "Recognized": color = "lime"
        elif status == "Unknown": color = "orange"

        label_text = f"{identity}"
        if status == "Recognized" and confidence is not None:
            label_text += f" ({confidence:.2f})"
        elif status == "Unknown" and result_data.get("closest_match") and confidence is not None: 
            # MLP en yüksek olasılıklı sınıfı verir, eşiğin altındaysa "closest_match" olarak kullanılabilir
            label_text += f" ({result_data['closest_match']}?:{confidence:.2f})"
        elif status == "NoEmbedding":
            label_text = "Ozellik Yok"
        elif status == "Error" or status == "NoClassifier": # NoIndex yerine NoClassifier
            label_text = "Hata/Model Yok"
        
        font_size = max(12, int(frame_height / 45)); font = load_pil_font(size=font_size)
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=max(1, int(frame_height / 250)))
        try: text_bbox=draw.textbbox((x1,y1-font_size-2),label_text,font=font);text_w,text_h = text_bbox[2]-text_bbox[0],text_bbox[3]-text_bbox[1]
        except AttributeError: text_w,text_h = draw.textlength(label_text,font=font),font.getmask(label_text).size[1] if hasattr(font,'getmask') else font_size
        bg_y1=y1-text_h-4; bg_y1=y1+2 if bg_y1<0 else bg_y1
        draw.rectangle([(x1,bg_y1),(x1+text_w+4,bg_y1+text_h+2)],fill=color); draw.text((x1+2,bg_y1),label_text,fill="black",font=font)

    def recognize_faces_in_frame(self, frame_pil: Image.Image) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        annotated_frame = frame_pil.copy()
        draw = ImageDraw.Draw(annotated_frame)
        recognition_results: List[Dict[str, Any]] = []
        detected_faces = self.detector.detect_faces(frame_pil)

        for idx, face_info in enumerate(detected_faces):
            x1, y1, x2, y2 = face_info["box"]
            face_crop_pil = frame_pil.crop((x1, y1, x2, y2))

            if self.image_saver and \
               face_crop_pil.width >= self.config.MIN_FACE_SIZE_FOR_EMBEDDING and \
               face_crop_pil.height >= self.config.MIN_FACE_SIZE_FOR_EMBEDDING: # AppConfig yerine self.config
                try:
                    self.image_saver.schedule_save(face_crop_pil.copy(),
                        frame_number=self.current_frame_number_for_saving, detection_idx=idx)
                except Exception as e: logger.error(f"Yüz kaydı planlanırken hata: {e}")
            
            query_embedding = self.extractor.extract_embedding(face_crop_pil)

            current_identity = "Bilinmeyen"
            current_status = "NoEmbedding"
            current_confidence: Optional[float] = None
            # MLP en yüksek olasılıklı sınıfı döndürür, bu "closest_match" gibi düşünülebilir
            current_best_match_label: Optional[str] = None


            if query_embedding is not None:
                current_status = "Unknown" # Embedding var ama henüz tanınmadı
                if self.mlp_classifier and self.mlp_classifier.is_ready():
                    prediction = self.mlp_classifier.predict(query_embedding)
                    if prediction:
                        predicted_label, confidence = prediction
                        current_confidence = confidence
                        current_best_match_label = predicted_label # En iyi tahmin her zaman bu
                        
                        if confidence >= self.recognition_threshold: # MLP güven eşiği
                            current_identity = predicted_label
                            current_status = "Recognized"
                        # else: Güven düşük, kimlik "Bilinmeyen" kalır, status "Unknown" kalır
                    else:
                        current_status = "Error" # MLP tahmin hatası
                else:
                    current_status = "NoClassifier" # MLP modeli yüklenmemiş
            
            result_data = {
                "box": [x1, y1, x2, y2],
                "identity": current_identity,
                "distance": None, # MLP için distance yerine confidence kullanıyoruz
                "confidence": round(current_confidence, 4) if current_confidence is not None else None,
                "status": current_status,
                "closest_match": current_best_match_label, # En olası etiket
                "confidence_detection": face_info.get("confidence") 
            }
            recognition_results.append(result_data)
            self._draw_annotations(draw, frame_pil.height, result_data)
        
        self.current_frame_number_for_saving = None # Reset for next potential image call
        return annotated_frame, recognition_results

# MediaProcessor sınıfı büyük ölçüde aynı kalacak, sadece __init__ içinde config alabilir
class MediaProcessor:
    def __init__(self, recognition_service: FaceRecognitionService, output_base_dir: Path, config: AppConfig): # config eklendi
        self.recognition_service = recognition_service
        self.output_base_dir = output_base_dir
        self.config = config # config sakla
        # recognition_service artık kendi içinde AppConfig'e değil, parametrelerine bağımlı olmalı
        # FaceRecognitionService.__init__'e config parametresi ekleyip MIN_FACE_SIZE_FOR_EMBEDDING'i oradan alması daha iyi olur.
        # Şimdilik FaceRecognitionService.recognize_faces_in_frame içinde self.config.MIN_FACE_SIZE_FOR_EMBEDDING kullanıyoruz,
        # Bu yüzden FRS'ye config'i de vermeliyiz.
        self.recognition_service.config = config # FRS'ye config'i ata


    # ... (_create_output_dir, _save_json_results, process_image, process_video kodları öncekiyle aynı)
    def _create_output_dir(self) -> Path:
        ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"); od=self.output_base_dir/ts; od.mkdir(parents=True,exist_ok=True)
        logger.info(f"Çıktı dizini: {od}"); return od
    def _save_json_results(self, od:Path, data:Dict[str,Any]):
        op=od/"recognition_results.json"
        try:
            with open(op,'w',encoding='utf-8') as f: json.dump(data,f,indent=4,ensure_ascii=False)
            logger.info(f"JSON kaydedildi: {op}")
        except Exception as e: logger.error(f"JSON kaydetme hatası: {e}")
    def process_image(self, image_path_str: str):
        image_path = Path(image_path_str); logger.info(f"Resim: {image_path}")
        output_dir = self._create_output_dir()
        try: img_pil = Image.open(image_path).convert("RGB")
        except Exception as e: logger.error(f"Resim {image_path} açılamadı: {e}"); return
        st=time.time(); annotated_img, results = self.recognition_service.recognize_faces_in_frame(img_pil); et=time.time()
        pt=round(et-st,2); logger.info(f"Resim işleme süresi: {pt:.2f}s")
        out_img_path = output_dir / f"processed_{image_path.name}"
        try: annotated_img.save(out_img_path); logger.info(f"İşlenmiş resim: {out_img_path}")
        except Exception as e: logger.error(f"İşlenmiş resim kaydedilemedi: {e}")
        json_data = {"source":str(image_path.resolve()),"ts":datetime.datetime.now().isoformat(),
                     "threshold_type":"mlp_confidence", "threshold_value":self.recognition_service.recognition_threshold,
                     "duration_s":pt, "recognitions":results}
        self._save_json_results(output_dir, json_data)

    def process_video(self, video_path_str: str, process_every_n_frames: int):
        video_path = Path(video_path_str); logger.info(f"Video: {video_path}")
        output_dir = self._create_output_dir(); st_total=time.time()
        cap=cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): logger.error(f"Video {video_path} açılamadı."); return
        fps=cap.get(cv2.CAP_PROP_FPS); w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tf=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=30. if fps<=0 else fps
        logger.info(f"Video Detay: {w}x{h} @ {fps:.2f}FPS, Kareler: {tf if tf>0 else 'Bilinmiyor'}")
        out_vid_path = output_dir / f"processed_{video_path.stem}.mp4"
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'); out_writer:Optional[cv2.VideoWriter]=None
        try:
            out_writer=cv2.VideoWriter(str(out_vid_path),fourcc,fps,(w,h))
            if not out_writer.isOpened(): raise IOError("VideoWriter açılamadı.")
        except Exception as e: logger.error(f"VideoWriter ({out_vid_path}) hatası: {e}"); cap.release(); return
        
        frame_num, all_results = 0, []
        while True:
            ret,frame_cv2 = cap.read()
            if not ret: break
            frame_num+=1; print(f"\rİşlenen Kare: {frame_num}/{tf if tf>0 else '?'}",end="")
            if frame_num % process_every_n_frames == 0:
                if self.recognition_service.image_saver: self.recognition_service.set_current_frame_number(frame_num)
                frame_pil = Image.fromarray(cv2.cvtColor(frame_cv2,cv2.COLOR_BGR2RGB))
                annotated_pil, frame_recs = self.recognition_service.recognize_faces_in_frame(frame_pil)
                if frame_recs:
                    ts_ms=cap.get(cv2.CAP_PROP_POS_MSEC)
                    all_results.append({"frame":frame_num,"ts_ms":round(ts_ms) if ts_ms else None,"recognitions":frame_recs})
                out_cv2 = cv2.cvtColor(np.array(annotated_pil),cv2.COLOR_RGB2BGR)
            else: out_cv2 = frame_cv2
            if out_writer: out_writer.write(out_cv2)
        print("\nVideo işleme tamamlandı."); cap.release(); 
        if out_writer: out_writer.release()
        et_total=time.time(); total_dur=round(et_total-st_total,2)
        logger.info(f"Toplam video işleme süresi: {total_dur:.2f}s"); logger.info(f"İşlenmiş video: {out_vid_path}")
        json_data = {"source":str(video_path.resolve()),"ts":datetime.datetime.now().isoformat(),
                     "threshold_type":"mlp_confidence","threshold_value":self.recognition_service.recognition_threshold,
                     "duration_s":total_dur,"total_frames":frame_num, "analyzed_frames":frame_num//process_every_n_frames if process_every_n_frames>0 else frame_num,
                     "frames_output":len(all_results),"frame_details":all_results}
        self._save_json_results(output_dir, json_data)


def main_application():
    input_media_path = "sample.webm"  # Test edilecek resim veya video dosyasının yolu

    if not Path(input_media_path).exists():
        logger.error(f"Giriş medyası bulunamadı: {input_media_path}")
        return

    config = AppConfig() # config nesnesini burada oluştur
    
    # ImageSaver yolu güncellendi
    image_saver_instance = ImageSaver(base_save_dir=config.DETECTED_FACES_SAVE_DIR) 
    
    try:
        face_detector = FaceDetector(
            model_path=config.YOLO_FACE_MODEL_PATH,
            confidence_threshold=config.YOLO_CONFIDENCE_THRESHOLD
        )
        embedding_extractor = EmbeddingExtractor( # min_face_size config'den alınacak
            model_name=config.EMBEDDING_MODEL_NAME,
            detector_backend=config.DEEPFACE_DETECTOR_BACKEND_FOR_CROPS,
            min_face_size= config.MIN_FACE_SIZE_FOR_EMBEDDING # config'den eklendi
        )
        # FaissDatabase yerine TrainedMLPClassifier yüklenecek
        mlp_classifier = TrainedMLPClassifier(
            model_path=config.MLP_MODEL_PATH,
            encoder_path=config.MLP_ENCODER_PATH
        )
        if not mlp_classifier.is_ready():
            logger.critical("MLP Sınıflandırıcı modeli yüklenemedi. Lütfen modelin doğru yolda olduğundan ve eğitilmiş olduğundan emin olun.")
            if image_saver_instance: image_saver_instance.stop(wait_for_completion=False)
            return

        recognition_service = FaceRecognitionService(
            detector=face_detector,
            extractor=embedding_extractor,
            mlp_classifier=mlp_classifier, # FaissDatabase yerine MLP
            recognition_threshold=config.MLP_CONFIDENCE_THRESHOLD, # MLP için güven eşiği
            image_saver=image_saver_instance
        )
        # recognition_service.config = config # FRS'nin config'e erişimi için (eğer MIN_FACE_SIZE gibi değerleri oradan alıyorsa)

        media_processor = MediaProcessor( # config parametresi eklendi
            recognition_service=recognition_service,
            output_base_dir=config.RESULTS_BASE_DIR,
            config=config # config eklendi
        )
        
        # FRS.recognize_faces_in_frame içindeki AppConfig.MIN_FACE_SIZE_FOR_EMBEDDING
        # çağrısını self.config.MIN_FACE_SIZE_FOR_EMBEDDING yapmak için FRS'ye config'i verdik.
        # Bu satırı MediaProcessor.__init__ içinde ekledim: self.recognition_service.config = config


        file_ext = Path(input_media_path).suffix.lower()
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']

        if file_ext in image_extensions:
            media_processor.process_image(input_media_path)
        elif file_ext in video_extensions:
            media_processor.process_video(input_media_path, config.VIDEO_PROCESS_EVERY_N_FRAMES)
        else:
            logger.error(f"Desteklenmeyen dosya uzantısı: {file_ext}")
            # ... (boş çıktı dizini silme mantığı aynı kalabilir) ...
            output_dir_to_clean = media_processor._create_output_dir() 
            if not any(output_dir_to_clean.iterdir()):
                try: output_dir_to_clean.rmdir()
                except OSError as e: logger.warning(f"Boş çıktı dizini silinemedi: {output_dir_to_clean} - {e}")

    except (FileNotFoundError, RuntimeError, ValueError) as e:
        logger.critical(f"Uygulama hatası: {e}", exc_info=True)
    except Exception as e:
        logger.critical(f"Beklenmedik genel hata: {e}", exc_info=True)
    finally:
        if 'image_saver_instance' in locals() and image_saver_instance:
            image_saver_instance.stop()

if __name__ == '__main__':
    main_application()