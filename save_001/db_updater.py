import datetime
import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Set
from collections import defaultdict
import cv2
import faiss
import numpy as np
from PIL import Image, UnidentifiedImageError, ImageOps
from deepface import DeepFace
from sklearn.cluster import DBSCAN
import mediapipe as mp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("project-02/yenidenemem_few_shot/face_teslim/face_data_manager_v_complete.log", mode='a')
    ],
    
)
logger = logging.getLogger(__name__)

class Config:
    BASE_DB_DIR = Path("project-02/yenidenemem_few_shot/face_teslim/face_db") 
    HUMANS_DIR_NAME = "Human"    
    
    DETECTED_SUBDIR_NAME = "detected" 
    CLASSES_SUBDIR_NAME = "Classes"   
    
    PRODUCTION_FAISS_INDEX_FILENAME = "face_index_v5_facenet512.index"
    PRODUCTION_FAISS_MAP_FILENAME = "face_index_map_v5.json"      
    
    TEMP_DATA_SUBDIR_NAME = "temp_manager_data" 
    WORKING_FAISS_INDEX_FILENAME = "working_face_index.index"
    WORKING_FAISS_MAP_FILENAME = "working_face_map.json"
        
    KNOWN_MATCH_THRESHOLD = 0.30 
    CLASS_MERGE_THRESHOLD = 0.40
    DBSCAN_EPS = 0.25
    DBSCAN_MIN_SAMPLES = 5
    MIN_FACES_FOR_NEW_CLASS_VIA_DBSCAN = 4
    
    ENABLE_FQA_MEDIAPIPE_CHECK: bool = True 
    MIN_MEDIAPIPE_FACE_CONFIDENCE: float = 0.5 
    BLURRINESS_THRESHOLD: float = 120.0 
    ENABLE_FQA_ILLUMINATION_CHECK = True
    MIN_BRIGHTNESS_THRESHOLD: int = 40
    MAX_BRIGHTNESS_THRESHOLD: int = 215

  
    FROM_DETECTED_SUBDIR_NAME = "from_detected" 
    IMAGES_SUBDIR_FOR_NUMERIC_CLASSES = "images" 
    REFERENCE_IMAGES_SUBDIR_FOR_NAMED = "reference_images" 
    NOISE_SUBDIR_NAME = "noise_or_unassigned"

    EMBEDDING_MODEL_NAME = 'Facenet512'
    DEEPFACE_DETECTOR_BACKEND_FOR_CROPS = 'skip'
    MIN_FACE_SIZE_FOR_PROCESSING = 30 
    STARTING_NUMERIC_ID_FOR_NEW_CLASSES = 1000 
    
    AUGMENT_FOR_WORKING_FAISS = True 
    AUGMENT_FOR_PRODUCTION_FAISS = True

    def __init__(self):
        self.HUMANS_DIR = self.BASE_DB_DIR / self.HUMANS_DIR_NAME
        self.DETECTED_DIR = self.HUMANS_DIR / self.DETECTED_SUBDIR_NAME
        self.CLASSES_DIR = self.HUMANS_DIR / self.CLASSES_SUBDIR_NAME 
        
        self.PRODUCTION_FAISS_INDEX_PATH = self.HUMANS_DIR / self.PRODUCTION_FAISS_INDEX_FILENAME
        self.PRODUCTION_FAISS_MAP_PATH = self.HUMANS_DIR / self.PRODUCTION_FAISS_MAP_FILENAME
        
        self.TEMP_DATA_DIR = self.HUMANS_DIR / self.TEMP_DATA_SUBDIR_NAME
        self.WORKING_FAISS_INDEX_PATH = self.TEMP_DATA_DIR / self.WORKING_FAISS_INDEX_FILENAME
        self.WORKING_FAISS_MAP_PATH = self.TEMP_DATA_DIR / self.WORKING_FAISS_MAP_FILENAME
        
        self.NOISE_DIR = self.HUMANS_DIR / self.NOISE_SUBDIR_NAME

        self.HUMANS_DIR.mkdir(parents=True, exist_ok=True)
        self.DETECTED_DIR.mkdir(parents=True, exist_ok=True)
        self.CLASSES_DIR.mkdir(parents=True, exist_ok=True) 
        self.NOISE_DIR.mkdir(parents=True, exist_ok=True)
        self.TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)

def normalize_vector(v:np.ndarray): n=np.linalg.norm(v); return np.zeros_like(v).astype('f') if n<1e-9 else (v/n).astype('f')
def get_next_numeric_id(classes_dir: Path, starting_id: int) -> int:
    max_id = starting_id - 1; init_found = False
    if classes_dir.exists():
        for item in classes_dir.iterdir():
            if item.is_dir() and item.name.isdigit():
                try:cid=int(item.name); max_id=max(max_id,cid); init_found=True if cid>=starting_id else init_found
                except ValueError: pass
    return starting_id if not init_found else max_id + 1

class EmbeddingExtractor: # Kısaltılmış hali, öncekiyle aynı mantık
    def __init__(self, model_name: str, detector_backend: str, min_face_size: int):
        self.model_name,self.detector_backend,self.min_face_size=model_name,detector_backend,min_face_size
        self.emb_dim:int=-1; self._init_deepface()
        logger.info(f"Extractor: {self.model_name} (Min:{self.min_face_size}px, Dim:{self.emb_dim})")
    def _init_deepface(self):
        try:
            res=DeepFace.represent(np.zeros((64,64,3),dtype=np.uint8), model_name=self.model_name, enforce_detection=False, detector_backend=self.detector_backend, align=False)
            self.emb_dim = len(res[0]["embedding"]) if isinstance(res,list) and res and "embedding" in res[0] else {'Facenet512':512}.get(self.model_name,512)
        except Exception as e: logger.error(f"DeepFace modeli '{self.model_name}' yüklenemedi: {e}"); raise RuntimeError(f"DeepFace yüklenemedi: {e}")
    def get_embedding(self, img_path: Path) -> Optional[np.ndarray]:
        try:
            img=Image.open(img_path).convert("RGB")
            if img.width<self.min_face_size or img.height<self.min_face_size: return None
            res=DeepFace.represent(np.array(img),model_name=self.model_name,enforce_detection=False,detector_backend=self.detector_backend,align=False)
            if isinstance(res,list) and res and "embedding" in res[0] and len(res[0]["embedding"])==self.emb_dim: return np.array(res[0]["embedding"])
            return None
        except Exception: return None # Hata detayları ana döngüde loglanabilir

class FaissDBHandler: # Kısaltılmış hali, öncekiyle aynı mantık
    def __init__(self, emb_dim: int): self.emb_dim = emb_dim
    def load_index(self, idx_path:Path, map_path:Path) -> Tuple[Optional[faiss.Index], List[str]]:
        idx,id_map=None,[]
        if not idx_path.exists() or not map_path.exists(): return idx,id_map
        try:
            with open(map_path,'r',encoding='utf-8') as f: id_map=json.load(f)
            idx=faiss.read_index(str(idx_path))
            if idx.d!=self.emb_dim or idx.ntotal!=len(id_map): return None,[]
        except Exception: return None,[]
        return idx,id_map
    def create_and_save_index(self, embs:np.ndarray, ids:List[str], idx_path:Path, map_path:Path) -> bool:
        if embs.shape[0]!=len(ids) or embs.shape[1]!=self.emb_dim: return False
        try:
            index=faiss.IndexFlatIP(self.emb_dim); index.add(np.array([normalize_vector(e) for e in embs]))
            faiss.write_index(index,str(idx_path));
            with open(map_path,'w',encoding='utf-8') as f: json.dump(ids,f,indent=4)
            logger.info(f"FAISS kaydedildi: {idx_path.name} ({index.ntotal} vektör), {map_path.name}")
            return True
        except Exception as e: logger.error(f"FAISS kaydetme hatası: {e}"); return False
    def search_closest(self, idx:Optional[faiss.Index], id_map:List[str], query_emb:np.ndarray) -> Optional[Tuple[float,str]]:
        if not idx or idx.ntotal==0: return None
        try:
            D,I=idx.search(np.array([normalize_vector(query_emb)]),1)
            if I.size>0 and I[0][0]!=-1 and 0<=I[0][0]<len(id_map): return 1.-max(0.,min(1.,D[0][0])),id_map[I[0][0]]
            return None
        except Exception: return None


def augment_image(image_pil: Image.Image) -> List[Image.Image]:
    """Basit augmentasyonlar uygular."""
    augmented_images = [image_pil] 
    augmented_images.append(ImageOps.mirror(image_pil))

    return augmented_images



class FaceDataManager:
    def __init__(self, config: Config):
        self.config = config
        self.extractor = EmbeddingExtractor(config.EMBEDDING_MODEL_NAME, config.DEEPFACE_DETECTOR_BACKEND_FOR_CROPS, config.MIN_FACE_SIZE_FOR_PROCESSING)
        if self.extractor.emb_dim == -1: raise RuntimeError("EmbeddingExtractor başlatılamadı.")
        self.faiss_handler = FaissDBHandler(self.extractor.emb_dim)
        self.stats = defaultdict(int)
        self.mp_face_mesh = None
        if self.config.ENABLE_FQA_MEDIAPIPE_CHECK: # Yeni config parametresini kullanın
            try:
                self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True, 
                    max_num_faces=1,        
                    refine_landmarks=False, # Temel landmarklar FQA için yeterli olabilir
                    min_detection_confidence=self.config.MIN_MEDIAPIPE_FACE_CONFIDENCE
                )
                logger.info("MediaPipe Face Mesh modeli başarıyla yüklendi.")
            except Exception as e:
                logger.error(f"MediaPipe Face Mesh yüklenemedi: {e}. MediaPipe FQA atlanacak.")
                self.mp_face_mesh = None
        
        logger.info(f"FaceDataManager başlatıldı (FQA için MediaPipe {'aktif' if self.mp_face_mesh else 'devre dışı'}).")


    def _move_file(self, src_path: Path, target_dir: Path):
        try:
            if not src_path.exists(): logger.warning(f"Kaynak dosya bulunamadı: {src_path}"); return False
            target_dir.mkdir(parents=True, exist_ok=True)
            fn = src_path.name
            dest = target_dir / fn
            c=0; base,sfx=dest.stem,dest.suffix
            while dest.exists(): c+=1; dest=target_dir/f"{base}_{c}{sfx}" 
            shutil.move(str(src_path), str(dest)); logger.debug(f"'{src_path.name}'->'{dest}' taşındı."); return True
        except Exception as e: logger.error(f"'{src_path.name}'->'{target_dir}' taşıma hatası: {e}"); return False

    def _get_class_image_paths(self, class_id_str: str) -> List[Path]:
        """Verilen bir sınıf ID'si için tüm geçerli resim yollarını toplar."""
        class_path = self.config.CLASSES_DIR / class_id_str
        image_paths: List[Path] = []
        
        source_subdirs = [class_path] 
        if not class_id_str.isdigit(): 
            source_subdirs.append(class_path / self.config.REFERENCE_IMAGES_SUBDIR_FOR_NAMED)
            source_subdirs.append(class_path / self.config.FROM_DETECTED_SUBDIR_NAME)
        else: 
             source_subdirs.append(class_path / self.config.IMAGES_SUBDIR_FOR_NUMERIC_CLASSES)

        processed_files: Set[Path] = set()
        for subdir in source_subdirs:
            if subdir.exists() and subdir.is_dir():
                for item in subdir.iterdir():
                    if item.is_file() and item.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        resolved_item = item.resolve()
                        if resolved_item not in processed_files:
                            image_paths.append(item) 
                            processed_files.add(resolved_item)
        if not image_paths:
            logger.warning(f"'{class_id_str}' sınıfı için hiç resim bulunamadı (beklenen yerlerde).")
        return image_paths


    def _is_face_quality_sufficient(self, image_pil: Image.Image, image_path_for_log: Path) -> bool:
        """Bir yüz kırpıntısının genel kalitesini değerlendirir (MediaPipe ile)."""
        self.stats['FQA_total_checked'] += 1
        
        # 1. Boyut Kontrolü (Mevcut)
        if image_pil.width < self.config.MIN_FACE_SIZE_FOR_PROCESSING or \
           image_pil.height < self.config.MIN_FACE_SIZE_FOR_PROCESSING:
            logger.debug(f"FQA Red (Boyut): '{image_path_for_log.name}' ({image_pil.size})")
            self.stats['FQA_rej_size'] += 1; return False

        if self._is_image_blurred(image_pil, self.config.BLURRINESS_THRESHOLD):
            logger.debug(f"FQA Red (Bulanık): '{image_path_for_log.name}'")
            self.stats['FQA_rej_blur'] += 1; return False

        # 3. Aydınlanma Kontrolü (Mevcut)
        if self.config.ENABLE_FQA_ILLUMINATION_CHECK:
            brightness = np.mean(np.array(image_pil.convert('L')))
            if not (self.config.MIN_BRIGHTNESS_THRESHOLD < brightness < self.config.MAX_BRIGHTNESS_THRESHOLD):
                logger.debug(f"FQA Red (Aydınlanma): '{image_path_for_log.name}' Parlaklık {brightness:.1f}")
                self.stats['FQA_rej_illumination'] += 1; return False
        
        # 4. MediaPipe Landmark/Yüz Varlığı Kontrolü (YENİ)
        if self.config.ENABLE_FQA_MEDIAPIPE_CHECK and self.mp_face_mesh:
            try:
                img_mp_rgb = np.array(image_pil.convert("RGB")) # MediaPipe RGB bekler
                results = self.mp_face_mesh.process(img_mp_rgb)
                
                if not results.multi_face_landmarks:
                    # MediaPipe kırpıntı içinde hiç yüz (ve dolayısıyla landmark) bulamadı.
                    logger.warning(f"FQA Red (MediaPipe): '{image_path_for_log.name}' - Kırpıntı içinde MediaPipe yüz/landmark bulamadı.")
                    self.stats['FQA_rej_mp_no_face_landmarks'] += 1
                    return False
                
                # Yüz bulundu, en azından temel bir kontrolü geçti.
                # İleri seviye: results.multi_face_landmarks[0].landmark listesindeki belirli
                # landmarkların varlığı veya pozisyonları daha detaylı kontrol edilebilir.
                # Şimdilik, landmarkların bulunması yeterli bir kalite göstergesi.
                logger.debug(f"FQA MediaPipe: '{image_path_for_log.name}' için landmarklar bulundu (Geçti).")
                self.stats['FQA_passed_mediapipe'] += 1

            except Exception as e_mp: # Geniş bir except bloğu, MediaPipe'ten gelebilecek hatalar için
                logger.error(f"FQA (MediaPipe): '{image_path_for_log.name}' işlenirken hata: {e_mp}")
                self.stats['FQA_rej_mp_exception'] += 1
                return False # Hata durumunda kalitesiz kabul et
        
        # logger.info(f"FQA Geçti: '{image_path_for_log.name}'") # Bu log çok sık olabilir, debug'a çekildi veya kaldırıldı.
        self.stats['FQA_passed_all_active_checks'] += 1
        return True

    def _is_image_blurred(self, image_pil: Image.Image, threshold: float) -> bool:
        """Görüntünün bulanık olup olmadığını Laplacian varyansı ile kontrol eder."""
        try:
            image_cv = np.array(image_pil.convert('L'))
            laplacian_var = cv2.Laplacian(image_cv, cv2.CV_64F).var()
            logger.debug(f"Bulanıklık Testi - Laplacian Varyansı: {laplacian_var:.2f} (Eşik: {threshold})")
            if laplacian_var < threshold:
                return True 
            return False 
        except Exception as e:
            logger.error(f"Bulanıklık kontrolü sırasında hata: {e}")
            return True 


    def _collect_embeddings_from_image_paths(self, image_paths: List[Path], class_id_for_all: str, 
                                             apply_augmentation: bool) -> Tuple[List[str], List[np.ndarray]]:
        """Verilen resim yollarından (ve augmentasyonlarından) embedding toplar."""
        cids, embs = [], []
        for img_path in image_paths:
            try:
                img_pil = Image.open(img_path).convert("RGB")
                images_to_process = augment_image(img_pil) if apply_augmentation else [img_pil]
                
                for aug_img_pil in images_to_process:
                   
                    img_np_for_deepface = np.array(aug_img_pil)
                    
                   
                    embedding_objs = DeepFace.represent(
                        img_path=img_np_for_deepface, 
                        model_name=self.config.EMBEDDING_MODEL_NAME,
                        enforce_detection=False,
                        detector_backend=self.config.DEEPFACE_DETECTOR_BACKEND_FOR_CROPS,
                        align=False
                    )
                    if embedding_objs and isinstance(embedding_objs, list) and len(embedding_objs) > 0 and "embedding" in embedding_objs[0]:
                        emb_vector = np.array(embedding_objs[0]["embedding"])
                        if emb_vector.shape[0] == self.extractor.emb_dim:
                            cids.append(class_id_for_all)
                            embs.append(emb_vector)
                        else: logger.warning(f"Augment edilmiş '{img_path.name}' için beklenmedik emb. boyutu.")
                    else: logger.warning(f"Augment edilmiş '{img_path.name}' için geçerli embedding yok.")
            except Exception as e:
                logger.error(f"'{img_path.name}' (veya augmentasyonu) işlenirken hata: {e}")
        return cids, embs

    def run_face_data_pipeline(self):
        logger.info("Yüz Veri Yönetim Pipelini Başlatılıyor...")
        self.stats.clear()
        
        logger.info("--- Aşama 0: 'detected' Klasöründeki Yüzler İçin Kalite Kontrolü ---")
        initial_detected_files = [f for f in self.config.DETECTED_DIR.iterdir() if f.is_file() and f.suffix.lower() in ['.png','.jpg','.jpeg']]
        valid_files_for_processing: List[Path] = []
        if not initial_detected_files: logger.info("'detected' klasöründe FQA için dosya yok.")
        else:
            logger.info(f"FQA için {len(initial_detected_files)} dosya bulundu.")
            for i, image_path in enumerate(initial_detected_files):
                logger.debug(f"[{i+1}/{len(initial_detected_files)}] FQA: {image_path.name}")
                try:
                    img_pil = Image.open(image_path).convert("RGB")
                    if self._is_face_quality_sufficient(img_pil, image_path): # GÜNCELLENMİŞ FQA METODU ÇAĞRILIYOR
                        valid_files_for_processing.append(image_path)
                    else: 
                        # _is_face_quality_sufficient False döndürdüyse, loglama ve stat kendi içinde yapıldı.
                        # Dosyayı gürültüye taşıyalım.
                        self._move_file(image_path, self.config.NOISE_DIR) 
                except UnidentifiedImageError:
                    logger.error(f"'{image_path.name}' geçerli bir görüntü değil veya bozuk (Aşama 0). Gürültüye taşınıyor.")
                    if self._move_file(image_path, self.config.NOISE_DIR): self.stats['A0_rej_corrupt_image'] += 1 # Yeni stat
                except Exception as e_fqa_loop: 
                    logger.error(f"'{image_path.name}' FQA döngüsünde hata: {e_fqa_loop}")
                    if self._move_file(image_path, self.config.NOISE_DIR): self.stats['A0_rej_fqa_loop_error'] += 1 # Yeni stat
        
        logger.info(f"Aşama 0 (FQA) tamamlandı. {len(valid_files_for_processing)} yüz sonraki aşamalara uygun.")
        logger.info(f"FQA İstatistikleri (Aşama 0): {dict(self.stats)}") # Güncel istatistikleri göster



        # --- Aşama 1: Kalan Yüzlerin İlk İşlenmesi (Production FAISS Kullanarak) ---
        logger.info("--- Aşama 1: Kalan Yüzlerin İşlenmesi (Production FAISS ile Eşleştirme) ---")
        prod_faiss_idx, prod_id_map = self.faiss_handler.load_index(
            self.config.PRODUCTION_FAISS_INDEX_PATH, 
            self.config.PRODUCTION_FAISS_MAP_PATH
        )

        unmatched_paths, unmatched_embs = [], []
        if prod_faiss_idx is None: 
            logger.warning("Production FAISS yüklenemedi, kaliteyi geçen tüm yüzler kümelenecek.")
            # Eğer production FAISS yoksa, kaliteyi geçen tüm yüzler unmatched olarak kabul edilir.
            for image_path in valid_files_for_processing: # Artık valid_files_for_processing üzerinden dönüyoruz
                embedding = self.extractor.get_embedding(image_path)
                if embedding is not None:
                    unmatched_paths.append(image_path)
                    unmatched_embs.append(embedding)
                    self.stats['A1_to_unmatched_no_prod_faiss'] += 1
                else: # Kaliteyi geçmişti ama embedding çıkarılamadı
                    logger.warning(f"'{image_path.name}' kaliteyi geçti ama embedding çıkarılamadı (Aşama 1). Gürültüye taşınıyor.")
                    if self._move_file(image_path, self.config.NOISE_DIR): self.stats['A1_to_noise_emb_fail_post_qc'] +=1
        else: # Production FAISS varsa normal eşleştirme
            total_valid_files = len(valid_files_for_processing)
            logger.info(f"Aşama 1'de {total_valid_files} kaliteyi geçmiş yüz işlenecek.")
            for i, image_path in enumerate(valid_files_for_processing): # Artık valid_files_for_processing üzerinden dönüyoruz
                logger.info(f"[{i+1}/{total_valid_files}] Aşama 1: {image_path.name}")
                embedding = self.extractor.get_embedding(image_path)
                if embedding is None:
                    logger.warning(f"'{image_path.name}' kaliteyi geçti ama embedding çıkarılamadı (Aşama 1). Gürültüye taşınıyor.")
                    if self._move_file(image_path, self.config.NOISE_DIR): self.stats['A1_to_noise_emb_fail_post_qc'] += 1
                    continue

                self.stats['A1_extracted_from_valid'] += 1
                match, matched_known = None, False
                match = self.faiss_handler.search_closest(prod_faiss_idx, prod_id_map, embedding)

                if match and match[0] < self.config.KNOWN_MATCH_THRESHOLD:
                    distance, identity_id = match
                    logger.info(f"Eşleşme (Prod): '{image_path.name}' -> {identity_id} (Uzaklık:{distance:.4f})")
                    target_class_dir = self.config.CLASSES_DIR / identity_id 
                    target_dest_subdir = target_class_dir / self.config.FROM_DETECTED_SUBDIR_NAME
                    if self._move_file(image_path, target_dest_subdir): self.stats['A1_matched_moved']+=1
                    matched_known = True
                if not matched_known: 
                    unmatched_paths.append(image_path); unmatched_embs.append(embedding)

        logger.info(f"Aşama 1 tamamlandı. {self.stats.get('A1_matched_moved',0)} bilinenle eşleşti. {len(unmatched_paths)} kümelemeye kaldı.")


        # --- Aşama 2: Kalanları Kümele ve Yeni Sayısal ID Sınıf Klasörleri Oluştur ---
        created_numeric_class_ids_this_run: List[str] = []
        if unmatched_embs:
            logger.info(f"--- Aşama 2: Kümeleme ve Yeni Sayısal Sınıflar ({len(unmatched_embs)} yüz) ---")
            embeddings_np = np.array(unmatched_embs)
            if embeddings_np.shape[0] >= self.config.DBSCAN_MIN_SAMPLES:
                norm_embs_dbscan = np.array([normalize_vector(e) for e in embeddings_np])
                dbscan = DBSCAN(eps=self.config.DBSCAN_EPS, min_samples=self.config.DBSCAN_MIN_SAMPLES, metric='cosine', n_jobs=-1)
                try:
                    clabels = dbscan.fit_predict(norm_embs_dbscan)
                    next_id_val = get_next_numeric_id(self.config.CLASSES_DIR, self.config.STARTING_NUMERIC_ID_FOR_NEW_CLASSES)
                    for ulabel in set(clabels):
                        c_indices = np.where(clabels == ulabel)[0]
                        c_paths = [unmatched_paths[i] for i in c_indices]
                        if ulabel == -1: # Gürültü
                            for pth in c_paths: 
                                if self._move_file(pth, self.config.NOISE_DIR): self.stats['A2_dbscan_noise']+=1
                            continue
                        if len(c_indices) >= self.config.MIN_FACES_FOR_NEW_CLASS_VIA_DBSCAN:
                            nid_str = str(next_id_val); next_id_val+=1
                            created_numeric_class_ids_this_run.append(nid_str)
                            # Yeni sınıf humans/classes/<nid_str>/images/ altına
                            tdir = self.config.CLASSES_DIR / nid_str / self.config.IMAGES_SUBDIR_FOR_NUMERIC_CLASSES
                            logger.info(f"Yeni sayısal sınıf '{nid_str}' ({len(c_indices)} yüz) -> {tdir}")
                            self.stats['A2_new_num_classes']+=1
                            for pth in c_paths: 
                                if self._move_file(pth, tdir): self.stats['A2_faces_in_new_num_classes']+=1
                        else: # Küçük kümeler
                            for pth in c_paths: 
                                if self._move_file(pth, self.config.NOISE_DIR): self.stats['A2_small_cluster_to_noise']+=1
                except Exception as e: 
                    logger.error(f"DBSCAN/küme işleme hatası: {e}. Kalanlar gürültüye taşınacak.")
                    for pth in unmatched_paths: # Hata durumunda kalanları (eğer varsa) gürültüye taşı
                        if pth.exists() and self._move_file(pth, self.config.NOISE_DIR): self.stats['A2_error_to_noise']+=1
            else: # Kümeleme için az yüz
                for pth in unmatched_paths: 
                    if self._move_file(pth, self.config.NOISE_DIR): self.stats['A2_not_enough_for_cluster_to_noise']+=1
        logger.info("Aşama 2 tamamlandı.")

        # --- Aşama 3: Kapsamlı "Çalışma (Working) FAISS" İndeksinin Oluşturulması ---
        logger.info("--- Aşama 3: Çalışma FAISS İndeksi Oluşturuluyor ---")
        all_current_class_folders_for_work_faiss: List[Path] = []
        if self.config.CLASSES_DIR.exists():
            for item in self.config.CLASSES_DIR.iterdir(): # humans/classes/ altındaki tüm klasörler
                if item.is_dir(): # Bunlar Erenay, Furkan, 1001, 1002 vb. olmalı
                    all_current_class_folders_for_work_faiss.append(item)
        
        work_faiss_class_ids: List[str] = []
        work_faiss_embeddings: List[np.ndarray] = []
        
        if not all_current_class_folders_for_work_faiss: 
            logger.warning("Çalışma FAISS için aday sınıf (humans/classes/ altında) bulunamadı.")
        else:
            logger.info(f"Çalışma FAISS için {len(all_current_class_folders_for_work_faiss)} aday sınıf klasörü bulundu.")
            for class_dir_path in all_current_class_folders_for_work_faiss:
                class_id = class_dir_path.name
                image_paths_for_this_class = self._get_class_image_paths(class_id) # Bu metodun içinde loglama var
                if image_paths_for_this_class:
                    # Çalışma FAISS'i için augmentasyon
                    cids_temp, embs_temp = self._collect_embeddings_from_image_paths(
                        image_paths_for_this_class, 
                        class_id, 
                        self.config.AUGMENT_FOR_WORKING_FAISS
                    )
                    work_faiss_class_ids.extend(cids_temp)
                    work_faiss_embeddings.extend(embs_temp)
            
        working_faiss_idx = None # Geçici FAISS indeksi
        if work_faiss_embeddings:
            work_faiss_embs_np = np.array(work_faiss_embeddings)
            if self.faiss_handler.create_and_save_index(work_faiss_embs_np, work_faiss_class_ids, 
                                                        self.config.WORKING_FAISS_INDEX_PATH, 
                                                        self.config.WORKING_FAISS_MAP_PATH):
                self.stats['A3_work_faiss_vecs'] = work_faiss_embs_np.shape[0]
                self.stats['A3_work_faiss_classes'] = len(set(work_faiss_class_ids))
                working_faiss_idx, _ = self.faiss_handler.load_index(self.config.WORKING_FAISS_INDEX_PATH, self.config.WORKING_FAISS_MAP_PATH) # Tekrar yükle
            else: logger.error("Çalışma FAISS oluşturulamadı. Birleştirme atlanacak.")
        else: logger.warning("Çalışma FAISS için embedding toplanamadı. Birleştirme atlanacak.")
        logger.info("Aşama 3 tamamlandı.")

        # --- Aşama 4: Sınıflar Arası Benzerlik Kontrolü ve Birleştirme (Çalışma FAISS Temelli) ---
        # final_consolidated_class_image_map: Dict[str, List[Path]] -> Hedef Sınıf ID -> [Orijinal Resim Yolları]
        # Bu map, Aşama 5'te nihai production FAISS'i oluşturmak için kullanılacak resimleri tutar.
        # Başlangıçta, çalışma FAISS'indeki tüm sınıfların resimlerini içerir.
        # Birleştirme oldukça, kaynak sınıfın resimleri hedef sınıfın listesine eklenir.
        
        final_consolidated_class_image_map: Dict[str, List[Path]] = defaultdict(list)
        if working_faiss_idx and work_faiss_class_ids: # work_faiss_class_ids, çalışma FAISS'indeki tüm vektörlerin ID'lerini tutar
            logger.info("--- Aşama 4: Sınıf Birleştirme Başlatılıyor ---")
            
            # 1. Çalışma FAISS'indeki her benzersiz sınıf için temsili (ortalama) embedding hesapla
            temp_class_embeddings_for_avg: Dict[str, List[np.ndarray]] = defaultdict(list)
            for cid_w, emb_w in zip(work_faiss_class_ids, work_faiss_embeddings):
                temp_class_embeddings_for_avg[cid_w].append(emb_w)
            
            representative_embeddings: Dict[str, np.ndarray] = {} # Sınıf ID -> Ortalama Embedding
            for cid_w, embs_w_list in temp_class_embeddings_for_avg.items():
                if embs_w_list: representative_embeddings[cid_w] = np.mean(np.array(embs_w_list), axis=0)

            active_class_ids_for_consolidation = set(representative_embeddings.keys())
            merged_source_ids: Set[str] = set() # Bu ID'ler başka birine birleşti, artık kullanılmayacak

            # Başlangıçta tüm aktif sınıfların resimlerini final_consolidated_class_image_map'e ekle
            for class_id_active in active_class_ids_for_consolidation:
                 # _get_class_image_paths, o sınıfın TÜM resimlerini (ref, from_detected, images) getirir
                final_consolidated_class_image_map[class_id_active].extend(self._get_class_image_paths(class_id_active))


            # --- Öncelik 1: Sayısal ID'leri -> İsimli Sınıflara Birleştirme (İTERATİF ve GÜNCELLENMİŞ) ---
            logger.info("Gelişmiş İTERATİF Sayısal ID -> İsimli Sınıf Birleştirme Başlatılıyor...")

            # representative_embeddings: Dict[str, np.ndarray] -> Her sınıfın o anki ORTALAMA embedding'ini tutar
            # temp_class_embeddings_for_avg: Dict[str, List[np.ndarray]] -> Her sınıfın TÜM bireysel embedding'lerini tutar
            # merged_source_ids: Hangi kaynak ID'lerin birleştiğini takip eder.
            # final_consolidated_class_image_map: Resim yollarını yönetir. (Bu zaten yukarıda doldurulmuştu)

            made_a_merge_in_this_pass = True
            pass_counter = 0
            while made_a_merge_in_this_pass:
                pass_counter += 1
                made_a_merge_in_this_pass = False
                logger.info(f"Sayısal->İsimli Birleştirme Geçişi #{pass_counter}...")

                # Her geçişin başında, hala aktif olan (birleştirilmemiş) sayısal ID'leri al
                # representative_embeddings, birleştirme oldukça güncelleneceği için her seferinde .keys() ile alınır
                current_numeric_ids_to_process = sorted(
                    [cid for cid in representative_embeddings.keys() if cid.isdigit() and cid not in merged_source_ids],
                    key=int
                )
                
                current_named_ids_for_target = [
                    cid for cid in representative_embeddings.keys() if not cid.isdigit()
                    # İsimli sınıflar merged_source_ids'e eklenmez (hedef oldukları için)
                ]

                if not current_numeric_ids_to_process or not current_named_ids_for_target:
                    logger.info(f"Geçiş #{pass_counter}: Birleştirilecek sayısal veya hedef isimli sınıf kalmadı/bulunamadı.")
                    break # Ana while döngüsünden çık

                for num_id in current_numeric_ids_to_process:
                    # num_id zaten bir önceki iterasyonda veya bu geçişin önceki adımlarında 
                    # merged_source_ids'e eklenmiş olabilir, tekrar kontrol
                    if num_id in merged_source_ids: 
                        continue 
                    
                    # num_id için temsili embedding ve bireysel embedding listesi var mı kontrol et
                    if num_id not in representative_embeddings or not temp_class_embeddings_for_avg.get(num_id):
                        logger.warning(f"'{num_id}' için temsili embedding veya bireysel embedding listesi bulunamadı, atlanıyor (muhtemelen zaten birleşmiş).")
                        merged_source_ids.add(num_id) # Tekrar işlememek için, eğer bir şekilde kalmışsa
                        if num_id in representative_embeddings: del representative_embeddings[num_id]
                        if num_id in temp_class_embeddings_for_avg: del temp_class_embeddings_for_avg[num_id]
                        if num_id in final_consolidated_class_image_map: del final_consolidated_class_image_map[num_id]
                        continue
                        
                    emb_num_representative = representative_embeddings[num_id]
                    
                    best_target_named_id_for_this_num_id: Optional[str] = None
                    min_distance_to_named_for_this_num_id = float('inf')

                    for named_id in current_named_ids_for_target:
                        if named_id not in representative_embeddings or not temp_class_embeddings_for_avg.get(named_id):
                            # Bu durum, bir isimli sınıfın bir şekilde silinmesiyle olabilir (beklenmedik)
                            logger.warning(f"'{named_id}' (isimli) için temsili embedding veya bireysel embedding listesi bulunamadı, karşılaştırma atlanıyor.")
                            continue
                        
                        emb_named_representative = representative_embeddings[named_id]
                        
                        dist = 1.0 - np.dot(normalize_vector(emb_num_representative), normalize_vector(emb_named_representative))
                        
                        logger.debug(f"Geçiş #{pass_counter} KONTROL: Sayısal '{num_id}' vs İsimli '{named_id}'. Uzaklık: {dist:.4f}, Eşik: {self.config.CLASS_MERGE_THRESHOLD}")

                        if dist < min_distance_to_named_for_this_num_id:
                            min_distance_to_named_for_this_num_id = dist
                            best_target_named_id_for_this_num_id = named_id
                    
                    if best_target_named_id_for_this_num_id and min_distance_to_named_for_this_num_id < self.config.CLASS_MERGE_THRESHOLD:
                        target_named_id = best_target_named_id_for_this_num_id # Daha net bir isim
                        logger.info(f"BİRLEŞTİRME (Sayısal->İsimli, Geçiş #{pass_counter}): "
                                    f"Kaynak '{num_id}' -> Hedef '{target_named_id}' (Uzaklık: {min_distance_to_named_for_this_num_id:.4f})")
                        made_a_merge_in_this_pass = True

                        # 1. `final_consolidated_class_image_map`'i güncelle (resim yollarını hedefe aktar)
                        source_images_paths_from_map = []
                        if num_id in final_consolidated_class_image_map:
                            source_images_paths_from_map = final_consolidated_class_image_map.pop(num_id) 
                            final_consolidated_class_image_map[target_named_id].extend(source_images_paths_from_map)
                            logger.debug(f"'{num_id}' resimleri ({len(source_images_paths_from_map)} adet) '{target_named_id}' sınıfına `final_consolidated_class_image_map` içinde aktarıldı.")
                        else:
                            logger.warning(f"'{num_id}' için `final_consolidated_class_image_map`'te resim yolu bulunamadı! Dosya taşıma için _get_class_image_paths kullanılacak.")

                        # 2. `temp_class_embeddings_for_avg`'ı güncelle (bireysel embedding'leri hedefe aktar)
                        source_individual_embeddings_list = []
                        if num_id in temp_class_embeddings_for_avg:
                            source_individual_embeddings_list = temp_class_embeddings_for_avg.pop(num_id)
                            # Hedefte anahtar yoksa oluştur ve sonra ekle
                            if target_named_id not in temp_class_embeddings_for_avg:
                                temp_class_embeddings_for_avg[target_named_id] = []
                            temp_class_embeddings_for_avg[target_named_id].extend(source_individual_embeddings_list)
                            logger.debug(f"'{num_id}' bireysel embedding'leri ({len(source_individual_embeddings_list)} adet) '{target_named_id}' sınıfına `temp_class_embeddings_for_avg` içinde aktarıldı.")
                        
                        # 3. Hedef `target_named_id`'nin temsili (ortalama) embedding'ini YENİDEN HESAPLA
                        if temp_class_embeddings_for_avg.get(target_named_id): # Listenin varlığını ve boş olmadığını kontrol et
                            all_embeddings_for_target_named = np.array(temp_class_embeddings_for_avg[target_named_id])
                            if all_embeddings_for_target_named.size > 0: # Boş array değilse ortalama al
                                representative_embeddings[target_named_id] = np.mean(all_embeddings_for_target_named, axis=0)
                                logger.info(f"'{target_named_id}' (isimli) sınıfının temsili embedding'i güncellendi ({len(all_embeddings_for_target_named)} embedding ile).")
                            else: # Resim/embedding kalmadıysa (beklenmedik)
                                if target_named_id in representative_embeddings: del representative_embeddings[target_named_id]
                                logger.warning(f"'{target_named_id}' için birleştirme sonrası temsili embedding hesaplanamadı (liste boş).")
                        else: # Hedefin embedding listesi yoksa (çok beklenmedik)
                            if target_named_id in representative_embeddings: del representative_embeddings[target_named_id]
                            logger.error(f"'{target_named_id}' için `temp_class_embeddings_for_avg`'de girdi yok!")


                        # 4. Kaynak `num_id`'yi `representative_embeddings`'ten sil (eğer varsa)
                        if num_id in representative_embeddings:
                            del representative_embeddings[num_id]
                        
                        # 5. Dosya sisteminde taşıma ve kaynak klasörü silme
                        source_class_folder_on_disk = self.config.CLASSES_DIR / num_id
                        target_folder_for_images_on_disk = self.config.CLASSES_DIR / target_named_id / self.config.FROM_DETECTED_SUBDIR_NAME
                        target_folder_for_images_on_disk.mkdir(parents=True, exist_ok=True)
                        
                        # Kaynak klasördeki TÜM resimleri al (images/ veya direkt altında)
                        actual_images_to_move_from_disk = self._get_class_image_paths(num_id) 
                        moved_count_this_merge_on_disk = 0
                        for img_to_move in actual_images_to_move_from_disk:
                            if img_to_move.exists():
                                if self._move_file(img_to_move, target_folder_for_images_on_disk):
                                    moved_count_this_merge_on_disk += 1
                        self.stats['A4_merged_files_num_to_name'] += moved_count_this_merge_on_disk
                        
                        if source_class_folder_on_disk.exists():
                            try: 
                                shutil.rmtree(source_class_folder_on_disk) 
                                logger.info(f"'{source_class_folder_on_disk}' klasörü birleştirme sonrası diskten silindi.")
                            except Exception as e: 
                                logger.error(f"Birleştirilen '{source_class_folder_on_disk}' diskten silinirken hata: {e}")
                        
                        merged_source_ids.add(num_id) # Bu ID'nin birleştiğini global olarak işaretle
                        self.stats['A4_merged_num_to_name_classes'] += 1
                        
                        # Bir num_id birleştiği için, bu num_id ile ilgili işlemler bu geçiş için bitti.
                        # Döngünün başına dönerek güncellenmiş representative_embeddings ile
                        # bir sonraki uygun num_id için devam edilecek.
            
            if not made_a_merge_in_this_pass and pass_counter > 0 : # İlk geçiş değilse ve hiç birleştirme olmadıysa
                 logger.info(f"Geçiş #{pass_counter}'de hiç Sayısal->İsimli birleştirme yapılmadı. İterasyon sonlandırılıyor.")
            elif pass_counter == 0 and not made_a_merge_in_this_pass: # Hiç sayısal veya isimli sınıf yoksa buraya düşer
                 logger.info("Sayısal->İsimli birleştirme için uygun sınıf bulunamadı.")


            logger.info(f"Sayısal->İsimli Birleştirme {pass_counter} geçiş sonunda tamamlandı.")
            

            
         # --- Öncelik 2: Kalan Sayısal ID'leri -> Birbirleriyle Birleştirme (Bağlantılı Bileşenler ile) ---
            logger.info("İkinci Öncelik: Kalan Sayısal ID'li Sınıflar Birbirleriyle Birleştiriliyor (Bağlantılı Bileşenler Yaklaşımı)...")

            # İsimli sınıflara birleştirilmemiş ve hala `representative_embeddings` içinde olan sayısal ID'ler:
            candidate_numeric_ids_for_numeric_merge = sorted(
                [cid for cid in representative_embeddings.keys() if cid.isdigit()],
                key=int 
            ) # merged_source_ids ile tekrar kontrol etmeye gerek yok, çünkü birleşenler 
              # representative_embeddings'ten zaten silinmiş olmalı (bir önceki Öncelik 1 adımında).

            if len(candidate_numeric_ids_for_numeric_merge) < 2:
                logger.info("Kendi aralarında birleştirilecek yeterli sayıda (en az 2) sayısal ID'li sınıf kalmadı.")
            else:
                logger.info(f"Sayısal-Sayısal birleştirme için {len(candidate_numeric_ids_for_numeric_merge)} aday ID var: {candidate_numeric_ids_for_numeric_merge}")
                
                # Disjoint Set Union (DSU) için parent map'i oluştur. Başlangıçta her ID kendi parent'ı.
                # Bu parent_map, birleştirme gruplarını yönetmemize yardımcı olacak.
                # Değerler, grubun "kök" veya "hedef" ID'si olacak.
                numeric_parent_map: Dict[str, str] = {cid: cid for cid in candidate_numeric_ids_for_numeric_merge}

                def find_set_root(class_id: str) -> str:
                    """Bir sınıf ID'sinin DSU'daki kök ebeveynini bulur (path compression ile)."""
                    if numeric_parent_map[class_id] == class_id:
                        return class_id
                    numeric_parent_map[class_id] = find_set_root(numeric_parent_map[class_id])
                    return numeric_parent_map[class_id]

                def union_sets(id1: str, id2: str):
                    """İki sınıfı (veya ait oldukları grupları) birleştirir.
                    Kural: Daha düşük olan sayısal ID kök/hedef olur.
                    """
                    root1 = find_set_root(id1)
                    root2 = find_set_root(id2)
                    if root1 != root2:
                        if int(root1) < int(root2): # root1 daha küçük, root2'yi ona bağla
                            numeric_parent_map[root2] = root1
                            logger.debug(f"DSU Birleştirme: '{root2}' -> '{root1}' (Artık aynı gruptalar)")
                        else: # root2 daha küçük (veya eşit, ama farklılar), root1'i ona bağla
                            numeric_parent_map[root1] = root2
                            logger.debug(f"DSU Birleştirme: '{root1}' -> '{root2}' (Artık aynı gruptalar)")
                        return True # Birleştirme yapıldı
                    return False # Zaten aynı gruptaydılar

                # Olası tüm sayısal ID çiftlerini karşılaştır ve DSU setlerini birleştir
                made_any_dsu_union = False
                for i in range(len(candidate_numeric_ids_for_numeric_merge)):
                    id_a = candidate_numeric_ids_for_numeric_merge[i]
                    # id_a için temsili embedding var mı kontrol et (normalde olmalı)
                    if id_a not in representative_embeddings: continue
                    emb_a = representative_embeddings[id_a]

                    for j in range(i + 1, len(candidate_numeric_ids_for_numeric_merge)):
                        id_b = candidate_numeric_ids_for_numeric_merge[j]
                        if id_b not in representative_embeddings: continue
                        emb_b = representative_embeddings[id_b]
                        
                        dist = 1.0 - np.dot(normalize_vector(emb_a), normalize_vector(emb_b))
                        logger.debug(f"Sayısal-Sayısal KONTROL: '{id_a}' vs '{id_b}'. Uzaklık: {dist:.4f}, Eşik: {self.config.CLASS_MERGE_THRESHOLD}")
                        if dist < self.config.CLASS_MERGE_THRESHOLD:
                            if union_sets(id_a, id_b):
                                made_any_dsu_union = True
                
                if not made_any_dsu_union:
                    logger.info("Sayısal ID'li sınıflar arasında DSU ile birleştirilecek çift bulunamadı.")
                else:
                    # DSU sonrası nihai birleştirme gruplarını oluştur
                    final_numeric_merge_groups: Dict[str, List[str]] = defaultdict(list)
                    for num_id_original in candidate_numeric_ids_for_numeric_merge:
                        root_id = find_set_root(num_id_original) # Herkesin nihai parent'ını bul
                        final_numeric_merge_groups[root_id].append(num_id_original)
                    
                    logger.info(f"Nihai sayısal ID birleştirme grupları: {dict(final_numeric_merge_groups)}")

                    # Birleştirmeleri uygula (dosya taşıma, map güncelleme vb.)
                    for target_root_id, source_ids_in_group in final_numeric_merge_groups.items():
                        if len(source_ids_in_group) < 2: # Tek elemanlı gruplar birleştirme gerektirmez
                            continue 
                        
                        # target_root_id bu grubun hedefi olacak.
                        # Diğer tüm source_id_in_group elemanları (target_root_id hariç) ona birleşecek.
                        logger.info(f"Sayısal Grup Birleştirme: Hedef '{target_root_id}', Kaynaklar: {[sid for sid in source_ids_in_group if sid != target_root_id]}")

                        # Hedef ID'nin (target_root_id) `final_consolidated_class_image_map` ve 
                        # `temp_class_embeddings_for_avg` içinde olduğundan emin ol.
                        if target_root_id not in final_consolidated_class_image_map:
                             final_consolidated_class_image_map[target_root_id] = self._get_class_image_paths(target_root_id)
                        if target_root_id not in temp_class_embeddings_for_avg:
                            # Bu durum, target_root_id'nin hiç embedding'i yoksa olabilir, bu yüzden tüm resimlerinden yeniden hesapla
                            target_images = self._get_class_image_paths(target_root_id)
                            target_embs_list = [self.extractor.get_embedding(p) for p in target_images if self.extractor.get_embedding(p) is not None]
                            if target_embs_list: temp_class_embeddings_for_avg[target_root_id] = target_embs_list


                        for source_id in source_ids_in_group:
                            if source_id == target_root_id: continue # Kendisiyle birleştirme

                            logger.info(f"  Birleştiriliyor (Sayısal->Sayısal): Kaynak '{source_id}' -> Hedef '{target_root_id}'")
                            
                            # 1. `final_consolidated_class_image_map` güncelle
                            if source_id in final_consolidated_class_image_map:
                                final_consolidated_class_image_map[target_root_id].extend(final_consolidated_class_image_map.pop(source_id))
                            
                            # 2. `temp_class_embeddings_for_avg` güncelle
                            if source_id in temp_class_embeddings_for_avg:
                                if target_root_id not in temp_class_embeddings_for_avg: temp_class_embeddings_for_avg[target_root_id] = []
                                temp_class_embeddings_for_avg[target_root_id].extend(temp_class_embeddings_for_avg.pop(source_id))

                            # 3. `representative_embeddings` güncelle (hedefin ortalamasını yeniden hesapla, kaynağı sil)
                            if temp_class_embeddings_for_avg.get(target_root_id):
                                all_embs_for_target = np.array(temp_class_embeddings_for_avg[target_root_id])
                                if all_embs_for_target.size > 0:
                                    representative_embeddings[target_root_id] = np.mean(all_embs_for_target, axis=0)
                            if source_id in representative_embeddings:
                                del representative_embeddings[source_id]

                            # 4. Dosya sisteminde taşıma ve kaynak klasörü silme
                            source_class_folder_on_disk = self.config.CLASSES_DIR / source_id
                            # Sayısal ID'li sınıfların resimleri IMAGES_SUBDIR_FOR_NUMERIC_CLASSES altında
                            target_images_folder_on_disk = self.config.CLASSES_DIR / target_root_id / self.config.IMAGES_SUBDIR_FOR_NUMERIC_CLASSES
                            target_images_folder_on_disk.mkdir(parents=True, exist_ok=True)
                            
                            images_to_move = self._get_class_image_paths(source_id)
                            moved_count_this_num_merge = 0
                            for img_to_move in images_to_move:
                                if img_to_move.exists():
                                    if self._move_file(img_to_move, target_images_folder_on_disk):
                                        moved_count_this_num_merge += 1
                            self.stats['A4_merged_files_num_to_num'] += moved_count_this_num_merge
                            
                            if source_class_folder_on_disk.exists():
                                try: shutil.rmtree(source_class_folder_on_disk)
                                except Exception as e: logger.error(f"Birleştirilen '{source_class_folder_on_disk}' silinirken hata: {e}")
                            
                            merged_source_ids.add(source_id) # Bu ID'nin birleştiğini global listeye ekle
                            self.stats['A4_merged_num_to_num_classes'] += 1
            
            logger.info("Sayısal ID'li sınıfların kendi aralarında birleştirilmesi tamamlandı.")
            # --- Geliştirilmiş Bağlantılı Bileşenler ile Sayısal ID Birleştirme Sonu ---

            
        logger.info("Aşama 4 (Sınıf Birleştirme) tamamlandı.")

        # --- Aşama 5: Nihai "Production FAISS" İndeksini Oluşturma ---
        logger.info("--- Aşama 5: Nihai Production FAISS İndeksi Oluşturuluyor ---")
        if not final_consolidated_class_image_map:
            logger.warning("Nihai FAISS oluşturmak için hiç konsolide sınıf bulunamadı.")
        else:
            logger.info(f"Nihai Production FAISS için {len(final_consolidated_class_image_map)} sınıf kullanılacak.")
            final_faiss_ids_prod: List[str] = []
            final_faiss_embeddings_prod: List[np.ndarray] = []

            for class_id, image_paths_for_class in final_consolidated_class_image_map.items():
                if not image_paths_for_class: 
                    logger.warning(f"'{class_id}' sınıfı için nihai FAISS'e eklenecek resim bulunamadı (birleştirme sonrası?).")
                    continue
                logger.debug(f"'{class_id}' için {len(image_paths_for_class)} resimden Production FAISS için embedding çıkarılıyor...")
                cids_temp, embs_temp = self._collect_embeddings_from_image_paths(
                    image_paths_for_class, 
                    class_id, 
                    self.config.AUGMENT_FOR_PRODUCTION_FAISS
                )
                final_faiss_ids_prod.extend(cids_temp)
                final_faiss_embeddings_prod.extend(embs_temp)
            
            if final_faiss_embeddings_prod:
                final_faiss_embs_np_prod = np.array(final_faiss_embeddings_prod)
                
                # Eski Production FAISS'i yedekle
                # ... (yedekleme kodu öncekiyle aynı) ...
                ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                pidx_path, pmap_path = self.config.PRODUCTION_FAISS_INDEX_PATH, self.config.PRODUCTION_FAISS_MAP_PATH
                if pidx_path.exists(): shutil.move(str(pidx_path), str(pidx_path.parent / f"{pidx_path.stem}_backup_{ts}{pidx_path.suffix}"))
                if pmap_path.exists(): shutil.move(str(pmap_path), str(pmap_path.parent / f"{pmap_path.stem}_backup_{ts}{pmap_path.suffix}"))

                if self.faiss_handler.create_and_save_index(final_faiss_embs_np_prod, final_faiss_ids_prod, pidx_path, pmap_path):
                    logger.info("Yeni Production FAISS indeksi başarıyla oluşturuldu.")
                    self.stats['A5_prod_faiss_vecs']=final_faiss_embs_np_prod.shape[0]; self.stats['A5_prod_faiss_classes']=len(set(final_faiss_ids_prod))
                else: logger.error("Yeni Production FAISS oluşturulamadı!")
            else: logger.error("Nihai Production FAISS için hiç embedding toplanamadı.")
        logger.info("Aşama 5 tamamlandı.")

        # --- Aşama 6: Temizlik ---
        logger.info("--- Aşama 6: 'detected' Klasörü ve Geçici Dosyaların Son Temizliği ---")
        # ... (Temizlik kodu öncekiyle aynı, WORKING_FAISS dosyalarını da sil) ...
        cleanup_count = 0
        if self.config.DETECTED_DIR.exists():
            for item in self.config.DETECTED_DIR.iterdir():
                if item.is_file():
                    if self._move_file(item, self.config.NOISE_DIR): cleanup_count +=1
        if self.config.WORKING_FAISS_INDEX_PATH.exists(): os.remove(self.config.WORKING_FAISS_INDEX_PATH)
        if self.config.WORKING_FAISS_MAP_PATH.exists(): os.remove(self.config.WORKING_FAISS_MAP_PATH)
        if self.config.TEMP_DATA_DIR.exists() and not any(self.config.TEMP_DATA_DIR.iterdir()):
            shutil.rmtree(self.config.TEMP_DATA_DIR)
        if cleanup_count > 0: self.stats['A6_final_cleanup'] = cleanup_count
        logger.info(f"Aşama 6 (Temizlik) tamamlandı.")
        
        logger.info("--- Yüz Veri Yöneticisi Tamamlandı ---"); logger.info("İstatistikler:")
        for key, value in sorted(self.stats.items()): logger.info(f"  {key}: {value}")

# --- Ana Çalıştırma Bloğu ---
if __name__ == "__main__":
    cfg = Config()
    try:
        manager = FaceDataManager(cfg)
        # Üretim FAISS dosyalarının varlığını kontrol et (Aşama 1 için gerekli)
        if not cfg.PRODUCTION_FAISS_INDEX_PATH.exists() or not cfg.PRODUCTION_FAISS_MAP_PATH.exists():
            logger.warning("Production FAISS index/map dosyaları bulunamadı.")
            logger.warning("Eğer bu ilk çalıştırma ise, boş bir Production FAISS oluşturulabilir veya script devam edip sadece yeni sınıflar oluşturabilir.")
            # Burada boş bir production FAISS oluşturma seçeneği eklenebilir veya kullanıcıya bırakılabilir.
            # Şimdilik, eğer yoksa, Aşama 1'de eşleştirme yapmadan devam edecek.
        
        manager.run_face_data_pipeline()

    except RuntimeError as e: logger.critical(f"Kritik çalışma zamanı hatası: {e}")
    except Exception as e: logger.critical(f"Beklenmedik genel bir hata oluştu: {e}", exc_info=True)