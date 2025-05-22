import datetime
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import cv2
import faiss
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from deepface import DeepFace
import queue
import threading
import uuid
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AppConfig:
    EMBEDDING_MODEL_NAME: str = 'Facenet512'
    YOLO_FACE_MODEL_PATH: str = 'project-02/yolov11m-face.pt' 
    DEEPFACE_DETECTOR_BACKEND_FOR_CROPS: str = 'skip'
    DEFAULT_RECOGNITION_THRESHOLD: float = 0.50
    FAISS_INDEX_FILE: str = 'project-02/yenidenemem_few_shot/face_teslim/face_db/Human/face_index_v5_facenet512.index'
    INDEX_MAP_FILE: str = 'project-02/yenidenemem_few_shot/face_teslim/face_db/Human/face_index_map_v5.json' 
    RESULTS_BASE_DIR: Path = Path("results_face_professional")
    YOLO_CONFIDENCE_THRESHOLD: float = 0.5
    MIN_FACE_SIZE_FOR_EMBEDDING: int = 25 
    VIDEO_PROCESS_EVERY_N_FRAMES: int = 4 
    HUMANS_BASE_DIR: Path = Path("project-02/yenidenemem_few_shot/face_teslim/face_db/Human")
    DETECTED_FACES_SUBDIR: str = "detected" 
    FONT_PATH_WINDOWS: str = "arial.ttf"
    FONT_PATH_LINUX: str = "DejaVuSans.ttf"
    DEFAULT_FONT_SIZE: int = 15

    @staticmethod
    def get_font_path() -> Optional[str]:
        font_path = AppConfig.FONT_PATH_WINDOWS if os.name == 'nt' else AppConfig.FONT_PATH_LINUX
        try:
            ImageFont.truetype(font_path, AppConfig.DEFAULT_FONT_SIZE)
            return font_path
        except IOError:
            logging.warning(f"Belirtilen font ('{font_path}') bulunamadı. Varsayılan font kullanılacak.")
            return None

def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """Normalizes a numpy vector for cosine similarity using inner product."""
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return np.zeros_like(vec).astype('float32')
    return (vec / norm).astype('float32')

def load_pil_font(size: int = 15) -> ImageFont.FreeTypeFont:
    """Loads the preferred font or falls back to default."""
    font_path = AppConfig.get_font_path()
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except IOError:
            pass
    try: # Pillow 10+
        return ImageFont.load_default(size=size)
    except TypeError: # Pillow < 10
        return ImageFont.load_default()


class ImageSaver:
    """
    Tespit edilen yüz görüntülerini asenkron olarak diske kaydeder.
    """
    def __init__(self, base_save_dir: Path):
        self.save_dir = base_save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.image_queue = queue.Queue(maxsize=100) # Kuyruk boyutunu ayarlayabilirsiniz
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        logging.info(f"ImageSaver başlatıldı. Kayıt dizini: {self.save_dir}")

    def _generate_filename(self, frame_number: Optional[int] = None, detection_idx: Optional[int] = None) -> str:
        """Kaydedilecek yüz için benzersiz bir dosya adı oluşturur."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        parts = ["face"]
        if frame_number is not None:
            parts.append(f"f{frame_number}")
        if detection_idx is not None:
            parts.append(f"d{detection_idx}")
        parts.append(timestamp)
        # Alternatif olarak UUID de eklenebilir: parts.append(str(uuid.uuid4().hex)[:8])
        return "_".join(parts) + ".png"

    def _worker(self):
        """Kuyruktan görüntüleri alıp diske yazan thread fonksiyonu."""
        while not self.stop_event.is_set() or not self.image_queue.empty():
            try:
                # Kuyruktan eleman almak için kısa bir timeout ile bekle,
                # böylece stop_event düzenli kontrol edilebilir.
                image_pil, frame_num, det_idx = self.image_queue.get(timeout=0.1)
                filename = self._generate_filename(frame_num, det_idx)
                filepath = self.save_dir / filename
                try:
                    image_pil.save(filepath)
                    logging.debug(f"Tespit edilen yüz kaydedildi: {filepath}") # Çok fazla log üretebilir
                except Exception as e:
                    logging.error(f"Yüz resmi {filepath} kaydedilemedi: {e}")
                finally:
                    self.image_queue.task_done() # İşlemin bittiğini işaretle
            except queue.Empty:
                # Kuyruk boşsa ve durdurma sinyali gelmediyse devam et
                if self.stop_event.is_set():
                    break 
            except Exception as e:
                logging.error(f"ImageSaver _worker hatası: {e}")


    def schedule_save(self, image_pil: Image.Image, frame_number: Optional[int] = None, detection_idx: Optional[int] = None):
        """Bir yüz görüntüsünün kaydedilmesi için kuyruğa ekler."""
        if self.stop_event.is_set():
            logging.warning("ImageSaver durduruluyor, yeni kayıt planlanamaz.")
            return
        try:
            # Kuyruğa eklerken bloklamayan put kullan ve kuyruk doluysa logla
            self.image_queue.put_nowait((image_pil, frame_number, detection_idx))
        except queue.Full:
            logging.warning(f"ImageSaver kuyruğu dolu. Yüz (kare:{frame_number}, idx:{detection_idx}) kaydedilemedi.")


    def stop(self, wait_for_completion: bool = True):
        """ImageSaver'ı durdurur ve bekleyen tüm kayıtların tamamlanmasını bekler."""
        logging.info("ImageSaver durduruluyor...")
        self.stop_event.set() # Durdurma sinyalini ayarla
        if wait_for_completion:
            self.image_queue.join() # Kuyruktaki tüm işlerin bitmesini bekle
        
        # Worker thread'in sonlanmasını bekle (kısa bir timeout ile)
        self.worker_thread.join(timeout=2.0) 
        if self.worker_thread.is_alive():
            logging.warning("ImageSaver worker thread'i zamanında durmadı.")
        logging.info("ImageSaver durduruldu.")


class FaceDetector:
    """Detects faces in an image using YOLO model."""
    def __init__(self, model_path: str, confidence_threshold: float):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            logging.error(f"YOLO modeli bulunamadı: {self.model_path}")
            raise FileNotFoundError(f"YOLO model file not found: {self.model_path}")
        try:
            self.model = YOLO(self.model_path)
            logging.info(f"YOLO modeli '{self.model_path}' başarıyla yüklendi.")
        except Exception as e:
            logging.error(f"YOLO modeli yüklenemedi: {e}")
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def detect_faces(self, image_pil: Image.Image) -> List[Dict[str, Any]]:
        """Detects faces and returns a list of detection dictionaries."""
        if self.model is None:
            logging.error("YOLO modeli yüklenmemiş.")
            return []
        
        detected_faces = []
        try:
            results = self.model.predict(image_pil, verbose=False, conf=self.confidence_threshold)
        except Exception as e:
            logging.error(f"YOLO ile yüz tespiti sırasında hata: {e}")
            return []

        if results and isinstance(results, list):
            for result_set in results:
                if not hasattr(result_set, 'boxes') or result_set.boxes is None:
                    continue
                for box_data in result_set.boxes:
                    if not (box_data.xyxy is not None and len(box_data.xyxy) > 0 and \
                            box_data.conf is not None and len(box_data.conf) > 0):
                        continue

                    confidence = float(box_data.conf[0])
                    # Confidence check already done by predict, but good for safety
                    # if confidence < self.confidence_threshold: 
                    #     continue

                    x1, y1, x2, y2 = map(int, box_data.xyxy[0].tolist())
                    if x1 < x2 and y1 < y2: # Valid box
                        detected_faces.append({
                            "box": [x1, y1, x2, y2],
                            "confidence": confidence
                        })
        return detected_faces


class EmbeddingExtractor:
    """Extracts face embeddings using DeepFace."""
    def __init__(self, model_name: str, detector_backend: str):
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.embedding_dimension: int = -1
        self._initialize()

    def _initialize(self):
        try:
            dummy_img_for_warmup = np.zeros((64, 64, 3), dtype=np.uint8)
            embedding_result = DeepFace.represent(
                dummy_img_for_warmup,
                model_name=self.model_name,
                enforce_detection=False, # We pass pre-cropped faces
                detector_backend=self.detector_backend
            )
            if isinstance(embedding_result, list) and len(embedding_result) > 0 and "embedding" in embedding_result[0]:
                self.embedding_dimension = len(embedding_result[0]["embedding"])
                logging.info(f"DeepFace embedding modeli '{self.model_name}' (Boyut: {self.embedding_dimension}) kullanıma hazır.")
            else:
                self.embedding_dimension = 512 # Fallback, should be derived
                logging.warning(f"DeepFace.represent'ten embedding boyutu alınamadı. Varsayılan {self.embedding_dimension} kullanılıyor.")
        except Exception as e:
            logging.error(f"DeepFace modeli '{self.model_name}' yüklenemedi/kullanılamadı: {e}")
            raise RuntimeError(f"Failed to load DeepFace model: {e}")

    def get_embedding_dimension(self) -> int:
        if self.embedding_dimension <= 0:
             raise RuntimeError("Embedding boyutu belirlenemedi.")
        return self.embedding_dimension

    def extract_embedding(self, face_crop_pil: Image.Image) -> Optional[np.ndarray]:
        if face_crop_pil.width < AppConfig.MIN_FACE_SIZE_FOR_EMBEDDING or \
           face_crop_pil.height < AppConfig.MIN_FACE_SIZE_FOR_EMBEDDING:
            return None

        try:
            face_crop_numpy_rgb = np.array(face_crop_pil.convert('RGB'))
            face_crop_numpy_bgr = cv2.cvtColor(face_crop_numpy_rgb, cv2.COLOR_RGB2BGR) 
            
            embedding_objs = DeepFace.represent(
                img_path=face_crop_numpy_bgr,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend=self.detector_backend
            )
            if embedding_objs and isinstance(embedding_objs, list) and len(embedding_objs) > 0:
                embedding = np.array(embedding_objs[0]["embedding"])
                if embedding.shape[0] != self.embedding_dimension:
                    logging.error(f"Embedding boyutu ({embedding.shape[0]}) != beklenen ({self.embedding_dimension}).")
                    return None
                return embedding
            return None
        except ValueError as ve:
            if "shape" in str(ve).lower():
                logging.debug(f"Embedding çıkarılırken boyut uyumsuzluğu: {ve}. Yüz: {face_crop_pil.size}")
            elif "could not be detected" in str(ve).lower():
                pass 
            else:
                logging.warning(f"Embedding çıkarılırken ValueError: {ve}")
            return None
        except Exception as e:
            err_str = str(e).lower()
            if "face could not be detected" not in err_str and "singleton array" not in err_str:
                logging.warning(f"Embedding çıkarılırken beklenmedik hata: {e}")
            return None


class FaissDatabase:
    """Manages FAISS index for face embeddings."""
    def __init__(self, index_path: str, map_path: str, expected_embedding_dim: int):
        self.index_path = Path(index_path)
        self.map_path = Path(map_path)
        self.expected_embedding_dim = expected_embedding_dim
        self.faiss_index: Optional[faiss.Index] = None
        self.index_to_name_map: List[str] = []
        self._load_database()

    def _load_database(self):
        logging.info("Mevcut FAISS veritabanı yükleniyor...")
        if self.expected_embedding_dim <= 0:
            logging.error("Embedding boyutu bilinmiyor. Veritabanı yüklenemez.")
            raise RuntimeError("Embedding dimension unknown, cannot load database.")

        if not self.map_path.exists():
            logging.error(f"FAISS index harita dosyası bulunamadı: {self.map_path}")
            raise FileNotFoundError(f"FAISS index map file not found: {self.map_path}")
        try:
            with open(self.map_path, 'r', encoding='utf-8') as f:
                self.index_to_name_map = json.load(f)
            logging.info(f"FAISS index haritası '{self.map_path}' yüklendi ({len(self.index_to_name_map)} girdi).")
        except Exception as e:
            logging.error(f"FAISS index haritası yüklenirken hata: {e}")
            raise RuntimeError(f"Failed to load FAISS index map: {e}")

        if not self.index_to_name_map:
            logging.error("FAISS index haritası boş. Tanıma yapılamaz.")
            raise ValueError("FAISS index map is empty.")

        if not self.index_path.exists():
            logging.error(f"FAISS index dosyası bulunamadı: {self.index_path}")
            raise FileNotFoundError(f"FAISS index file not found: {self.index_path}")
        try:
            self.faiss_index = faiss.read_index(str(self.index_path)) # faiss expects string path
            logging.info(f"FAISS index '{self.index_path}' yüklendi ({self.faiss_index.ntotal} vektör).")

            if self.faiss_index.ntotal != len(self.index_to_name_map):
                msg = f"FAISS index boyutu ({self.faiss_index.ntotal}) ile harita boyutu ({len(self.index_to_name_map)}) eşleşmiyor!"
                logging.error(msg)
                raise ValueError(msg)
            if self.faiss_index.d != self.expected_embedding_dim:
                msg = f"Yüklenen FAISS index boyutu ({self.faiss_index.d}) ile beklenen embedding boyutu ({self.expected_embedding_dim}) farklı!"
                logging.error(msg)
                raise ValueError(msg)
        except Exception as e:
            logging.error(f"FAISS index yüklenirken hata: {e}")
            raise RuntimeError(f"Failed to load FAISS index: {e}")
        logging.info("FAISS veritabanı başarıyla yüklendi.")

    def search(self, query_embedding: np.ndarray, k: int = 1) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return None, None
        
        normalized_query = normalize_vector(query_embedding)
        query_vector_2d = np.array([normalized_query]).astype('float32')
        
        try:
            distances, indices = self.faiss_index.search(query_vector_2d, k)
            return distances, indices
        except Exception as e:
            logging.error(f"FAISS arama hatası: {e}")
            return None, None

    def get_identity(self, index_in_db: int) -> Optional[str]:
        if 0 <= index_in_db < len(self.index_to_name_map):
            return self.index_to_name_map[index_in_db]
        logging.warning(f"FAISS geçersiz index: {index_in_db}, harita boyutu: {len(self.index_to_name_map)}")
        return None


class FaceRecognitionService:
    def __init__(self, detector: FaceDetector, extractor: EmbeddingExtractor,
                 database: FaissDatabase, recognition_threshold: float,
                 image_saver: Optional[ImageSaver] = None):
        self.detector = detector
        self.extractor = extractor
        self.database = database
        self.recognition_threshold = recognition_threshold
        self.image_saver = image_saver 
        self.current_frame_number_for_saving: Optional[int] = None 

    def set_current_frame_number(self, frame_number: int):
        """Video işlerken mevcut kare numarasını ayarlar (kayıt için)."""
        self.current_frame_number_for_saving = frame_number

    def _draw_annotations(self, draw: ImageDraw.Draw, frame_height: int, result_data: Dict[str, Any]):
        x1, y1, x2, y2 = result_data["box"]
        identity = result_data["identity"]
        status = result_data["status"]
        min_distance = result_data["distance"]
        best_match_name = result_data["closest_match"]
        
        color = "red"
        if status == "Recognized": color = "lime"
        elif status == "Unknown": color = "orange"


        label_text = f"{identity}"
        if status == "Recognized":
            label_text += f" ({min_distance:.2f})"
        elif status == "Unknown" and best_match_name is not None and min_distance is not None:
            label_text += f" ({best_match_name}?:{min_distance:.2f})"
        elif status == "NoEmbedding":
            label_text = "Ozellik Yok"
        elif status == "Error" or status == "NoIndex":
            label_text = "Hata"
        
        font_size = max(12, int(frame_height / 45))
        font = load_pil_font(size=font_size)

        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=max(1, int(frame_height / 250)))
        
        try: # Pillow 10+ textbbox
            text_bbox = draw.textbbox((x1, y1 - font_size - 2), label_text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        except AttributeError: # Fallback for older Pillow
            try: 
                text_w, text_h = draw.textlength(label_text, font=font), font.getbbox("A")[3] - font.getbbox("A")[1] # Approximate height
            except AttributeError: # Even older or different font object
                 text_w, text_h = len(label_text) * font_size * 0.6, font_size # Rough estimate

        bg_y1 = y1 - text_h - 4
        if bg_y1 < 0: bg_y1 = y1 + 2 # Adjust if text goes off screen top
        
        draw.rectangle([(x1, bg_y1), (x1 + text_w + 4, bg_y1 + text_h + 2)], fill=color)
        draw.text((x1 + 2, bg_y1), label_text, fill="black", font=font)

    def recognize_faces_in_frame(self, frame_pil: Image.Image) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        annotated_frame = frame_pil.copy()
        draw = ImageDraw.Draw(annotated_frame)
        recognition_results: List[Dict[str, Any]] = []

        detected_faces = self.detector.detect_faces(frame_pil)

        for idx, face_info in enumerate(detected_faces):
            x1, y1, x2, y2 = face_info["box"]
            face_crop_pil = frame_pil.crop((x1, y1, x2, y2))

            # --- Tespit edilen yüzü kaydetme mantığı (önceki adımdaki gibi) ---
            if self.image_saver and \
               face_crop_pil.width >= AppConfig.MIN_FACE_SIZE_FOR_EMBEDDING and \
               face_crop_pil.height >= AppConfig.MIN_FACE_SIZE_FOR_EMBEDDING:
                try:
                    # logging.info(f"FRS: Yüz kaydı planlanıyor. Kare: {self.current_frame_number_for_saving if self.current_frame_number_for_saving is not None else 'N/A (Resim)'}, Tespit Idx: {idx}")
                    self.image_saver.schedule_save(
                        face_crop_pil.copy(),
                        frame_number=self.current_frame_number_for_saving,
                        detection_idx=idx
                    )
                except Exception as e:
                    logging.error(f"Yüz kaydı planlanırken hata: {e}")
            
            query_embedding = self.extractor.extract_embedding(face_crop_pil)

            current_identity = "Bilinmeyen"
            current_status = "NoEmbedding" # query_embedding None ise bu durum geçerli olacak
            current_distance = float('inf')
            current_best_match = None

            if query_embedding is not None:
                # Özellik vektörü başarıyla çıkarıldı, şimdi tanımayı dene
                current_status = "Unknown" # Henüz kim olduğu bilinmiyor ama embedding var

                if self.database.faiss_index and self.database.faiss_index.ntotal > 0:
                    # Veritabanı yüklü ve içinde kayıt var
                    try:
                        # FaissDatabase.search metodu zaten normalizasyon yapıyor
                        # ve IndexFlatIP kullanılıyorsa iç çarpımı (inner product) döndürüyor.
                        faiss_inner_products, faiss_indices = self.database.search(query_embedding, k=1)

                        if faiss_indices is not None and faiss_inner_products is not None and \
                           faiss_indices.size > 0 and faiss_inner_products.size > 0:
                            
                            db_index = faiss_indices[0][0] 
                            inner_product = faiss_inner_products[0][0] 

                            # İç çarpımı (normalize vektörler için kosinüs benzerliğidir)
                            # kosinüs uzaklığına çevir (1 - benzerlik)
                            similarity = max(0.0, min(1.0, inner_product)) 
                            calculated_cosine_distance = 1.0 - similarity
                            current_distance = calculated_cosine_distance 

                            retrieved_name = self.database.get_identity(db_index)
                            if retrieved_name:
                                current_best_match = retrieved_name 
                                if current_distance < self.recognition_threshold:
                                    current_identity = retrieved_name 
                                    current_status = "Recognized"    
                                # else: (uzaklık eşikten büyükse)
                                # current_status "Unknown" olarak kalır,
                                # current_identity "Bilinmeyen" olarak kalır.
                            else:
                                # Veritabanındaki index haritasında bir sorun var
                                logging.warning(f"FRS: FAISS'ten alınan index ({db_index}) haritada bulunamadı.")
                                current_status = "Error" # Hata durumu
                        # else: FAISS araması sonuç döndürmedi (çok nadir bir durum, k=1 için)
                        # current_status "Unknown" olarak kalır.
                        
                    except Exception as e:
                        logging.error(f"FRS: FAISS arama sırasında bir hata oluştu: {e}")
                        current_status = "Error" 
                else:
                    # Veritabanı yüklenmemiş veya boş
                    current_status = "NoIndex"
            # else: (query_embedding is None ise)
            # current_status "NoEmbedding" olarak kalır.

            result_data = {
                "box": [x1, y1, x2, y2],
                "identity": current_identity,
                "distance": round(current_distance, 4) if current_distance != float('inf') else None,
                "status": current_status,
                "closest_match": current_best_match,
                "confidence_detection": face_info.get("confidence") 
            }
            
            recognition_results.append(result_data)
            self._draw_annotations(draw, frame_pil.height, result_data) 


        # Kare işlendikten sonra kare numarasını sıfırla (resim işleme için)
        # Eğer sadece video içinse bu sıfırlama gerekmeyebilir veya farklı yönetilebilir.
        # self.current_frame_number_for_saving = None 
        # Şimdilik bu satırı yorumda bırakıyorum, kullanım senaryonuza göre ayarlayın.
        # Eğer recognize_faces_in_frame hem resim hem video için kullanılıyorsa,
        # video işleyicinin her kare için set_current_frame_number'ı çağırması yeterli.

        return annotated_frame, recognition_results


class MediaProcessor:
    """Handles processing of images and videos for face recognition."""
    def __init__(self, recognition_service: FaceRecognitionService, output_base_dir: Path):
        self.recognition_service = recognition_service
        self.output_base_dir = output_base_dir

    def _create_output_dir(self) -> Path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.output_base_dir / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Çıktı dizini: {output_dir}")
        return output_dir

    def _save_json_results(self, output_dir: Path, data: Dict[str, Any]):
        output_json_path = output_dir / "recognition_results.json"
        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            logging.info(f"JSON sonuçları kaydedildi: {output_json_path}")
        except Exception as e:
            logging.error(f"HATA: JSON sonuçları kaydedilemedi: {e}")

    def process_image(self, image_path_str: str):
        image_path = Path(image_path_str)
        logging.info(f"Resim işleniyor: {image_path}")
        output_dir = self._create_output_dir()

        try:
            img_pil = Image.open(image_path).convert("RGB")
        except Exception as e:
            logging.error(f"HATA: Resim dosyası açılamadı: {image_path} - {e}")
            return

        start_time = time.time()
        annotated_img, results_list = self.recognition_service.recognize_faces_in_frame(img_pil)
        end_time = time.time()
        processing_time = round(end_time - start_time, 2)
        logging.info(f"Resim işleme süresi: {processing_time:.2f} saniye")

        output_image_filename = f"processed_{image_path.name}"
        output_image_path = output_dir / output_image_filename
        try:
            annotated_img.save(output_image_path)
            logging.info(f"İşlenmiş resim kaydedildi: {output_image_path}")
        except Exception as e:
            logging.error(f"HATA: İşlenmiş resim kaydedilemedi: {e}")

        json_data = {
            "source_file": str(image_path.resolve()),
            "processing_timestamp": datetime.datetime.now().isoformat(),
            "threshold": self.recognition_service.recognition_threshold,
            "duration_seconds": processing_time,
            "recognitions": results_list
        }
        self._save_json_results(output_dir, json_data)

    def process_video(self, video_path_str: str, process_every_n_frames: int):
        video_path = Path(video_path_str)
        logging.info(f"Video işleniyor: {video_path}")
        output_dir = self._create_output_dir()
        start_time_total = time.time()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.error(f"HATA: Video dosyası açılamadı: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0: fps = 30.0
        logging.info(f"Video: {width}x{height} @ {fps:.2f} FPS, Kare Sayısı: {total_frames if total_frames > 0 else 'Bilinmiyor'}")

        output_video_filename = f"processed_{video_path.stem}.mp4"
        output_video_path = output_dir / output_video_filename
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer: Optional[cv2.VideoWriter] = None
        try:
            out_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
            if not out_writer.isOpened(): raise IOError("VideoWriter açılamadı.")
        except Exception as e:
            logging.error(f"HATA: VideoWriter oluşturulamadı ({output_video_path}): {e}")
            cap.release()
            return

        frame_number = 0
        all_frame_results = []

        while True:
            ret, frame_cv2_bgr = cap.read()
            if not ret:
                break
            
            frame_number += 1
            if total_frames > 0:
                print(f"\rİşlenen Kare: {frame_number}/{total_frames}", end="")
            else:
                print(f"\rİşlenen Kare: {frame_number}", end="")

            if frame_number % process_every_n_frames == 0:
                # Yüz tanıma servisine mevcut kare numarasını bildir (kayıt için)
                if self.recognition_service.image_saver: # Sadece image_saver varsa kare no gönder
                    self.recognition_service.set_current_frame_number(frame_number)

                frame_pil_rgb = Image.fromarray(cv2.cvtColor(frame_cv2_bgr, cv2.COLOR_BGR2RGB))
                annotated_pil_frame, frame_recognition_list = self.recognition_service.recognize_faces_in_frame(frame_pil_rgb)
                
                # ... (geri kalan json'a ekleme ve video yazma mantığı aynı) ...
                if frame_recognition_list:
                    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    all_frame_results.append({
                        "frame_number": frame_number,
                        "timestamp_ms": round(timestamp_ms) if timestamp_ms is not None else None,
                        "recognitions": frame_recognition_list
                    })
                output_cv2_frame = cv2.cvtColor(np.array(annotated_pil_frame), cv2.COLOR_RGB2BGR)
            else:
                output_cv2_frame = frame_cv2_bgr
            
            if out_writer:
                out_writer.write(output_cv2_frame)
        
        print("\nVideo işleme tamamlandı.")
        cap.release()
        if out_writer: out_writer.release()
        
        end_time_total = time.time()
        total_duration = round(end_time_total - start_time_total, 2)
        logging.info(f"Toplam video işleme süresi: {total_duration:.2f} saniye")
        logging.info(f"İşlenmiş video kaydedildi: {output_video_path}")

        json_data = {
            "source_file": str(video_path.resolve()),
            "processing_timestamp": datetime.datetime.now().isoformat(),
            "threshold": self.recognition_service.recognition_threshold,
            "duration_seconds": total_duration,
            "processed_frames_total": frame_number,
            "frames_analyzed_for_recognition": frame_number // process_every_n_frames if process_every_n_frames > 0 else frame_number,
            "frames_with_recognitions_output": len(all_frame_results),
            "frames_recognition_details": all_frame_results
        }
        self._save_json_results(output_dir, json_data)


def main_application():
    #/home/earsal@ETE.local/Desktop/lightweight-human-pose-estimation-3d-demo.pytorch-master/selam02.mp4
    input_media_path = "/home/earsal@ETE.local/Downloads/trump-reu-2360131.jp"  


    if not Path(input_media_path).exists():
        logging.error(f"Giriş medyası bulunamadı: {input_media_path}")
        print(f"HATA: Lütfen 'input_media_path' değişkenini geçerli bir dosya yolu ile güncelleyin.")
        return

    config = AppConfig()
    detected_faces_save_path = config.HUMANS_BASE_DIR / config.DETECTED_FACES_SUBDIR
    image_saver_instance = ImageSaver(base_save_dir=detected_faces_save_path)
    
    try:
        face_detector = FaceDetector(
            model_path=config.YOLO_FACE_MODEL_PATH,
            confidence_threshold=config.YOLO_CONFIDENCE_THRESHOLD
        )

        embedding_extractor = EmbeddingExtractor(
            model_name=config.EMBEDDING_MODEL_NAME,
            detector_backend=config.DEEPFACE_DETECTOR_BACKEND_FOR_CROPS
        )
        embedding_dim = embedding_extractor.get_embedding_dimension()

        faiss_db = FaissDatabase(
            index_path=config.FAISS_INDEX_FILE,
            map_path=config.INDEX_MAP_FILE,
            expected_embedding_dim=embedding_dim
        )

        recognition_service = FaceRecognitionService(
            detector=face_detector,
            extractor=embedding_extractor,
            database=faiss_db,
            recognition_threshold=config.DEFAULT_RECOGNITION_THRESHOLD,
            image_saver=image_saver_instance 
        )

        media_processor = MediaProcessor(
            recognition_service=recognition_service,
            output_base_dir=config.RESULTS_BASE_DIR
        )

        file_ext = Path(input_media_path).suffix.lower()
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']

        if file_ext in image_extensions:
            media_processor.process_image(input_media_path)
        elif file_ext in video_extensions:
            media_processor.process_video(input_media_path, config.VIDEO_PROCESS_EVERY_N_FRAMES)
        else:
            logging.error(f"Desteklenmeyen dosya uzantısı: {file_ext}")
            output_dir_to_clean = media_processor._create_output_dir() # create to get the path
            if not any(output_dir_to_clean.iterdir()): # if empty
                try:
                    output_dir_to_clean.rmdir()
                except OSError as e:
                    logging.warning(f"Boş çıktı dizini silinemedi: {output_dir_to_clean} - {e}")


    except (FileNotFoundError, RuntimeError, ValueError) as e:
        logging.critical(f"Uygulama başlatılırken veya çalışırken kritik hata: {e}")
        print(f"KRİTİK HATA: {e}. Lütfen logları kontrol edin. Script sonlandırılıyor.")
    except Exception as e:
        logging.critical(f"Beklenmedik genel bir hata oluştu: {e}", exc_info=True)
        print(f"BEKLENMEDİK HATA: {e}. Lütfen logları kontrol edin. Script sonlandırılıyor.")
    finally:
        # Program sonlanmadan önce ImageSaver'ı durdur
        if 'image_saver_instance' in locals() and image_saver_instance:
            image_saver_instance.stop()


if __name__ == '__main__':
   
    main_application()