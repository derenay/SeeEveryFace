
import datetime
import json
import logging
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Set
from collections import defaultdict

import faiss
import numpy as np
from PIL import Image, UnidentifiedImageError 
from deepface import DeepFace

# --- Logging Yapılandırması ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("faiss_starter_no_augmentation.log", mode='a') # Log adı güncellendi
    ]
)
logger = logging.getLogger(__name__)

# --- Yapılandırma Sınıfı ---
class FaissStarterConfig:
    BASE_DB_DIR = Path("project-02/yenidenemem_few_shot/face_teslim/face_db") # face_data_manager.py ile tutarlı olmalı
    HUMANS_DIR_NAME = "Human"     # face_data_manager.py ile tutarlı olmalı
    CLASSES_SUBDIR_NAME = "Classes"   # Kaynak sınıfların olduğu yer
    
    PRODUCTION_FAISS_INDEX_FILENAME = "production_face_index.index" 
    PRODUCTION_FAISS_MAP_FILENAME = "production_face_map.json"       
    
    EMBEDDING_MODEL_NAME = 'Facenet512'
    DEEPFACE_DETECTOR_BACKEND_FOR_CROPS = 'skip'
    MIN_FACE_SIZE_FOR_PROCESSING = 30 
    
    # APPLY_AUGMENTATION_FOR_PROTOTYPES ve NUM_AUGMENTATIONS_PER_IMAGE kaldırıldı.

    def __init__(self):
        self.HUMANS_DIR = self.BASE_DB_DIR / self.HUMANS_DIR_NAME
        self.CLASSES_DIR = self.HUMANS_DIR / self.CLASSES_SUBDIR_NAME
        
        self.PRODUCTION_FAISS_INDEX_PATH = self.HUMANS_DIR / self.PRODUCTION_FAISS_INDEX_FILENAME
        self.PRODUCTION_FAISS_MAP_PATH = self.HUMANS_DIR / self.PRODUCTION_FAISS_MAP_FILENAME
        
        self.CLASSES_DIR.mkdir(parents=True, exist_ok=True)

# --- Yardımcı Fonksiyonlar ve Sınıflar ---
def normalize_vector(v:np.ndarray): n=np.linalg.norm(v); return np.zeros_like(v).astype('f') if n<1e-9 else (v/n).astype('f')

class EmbeddingExtractor:
    def __init__(self, model_name: str, detector_backend: str, min_face_size: int):
        self.model_name,self.detector_backend,self.min_face_size=model_name,detector_backend,min_face_size
        self.emb_dim:int=-1; self._init_deepface()
        logger.info(f"FAISS-Starter Extractor: {self.model_name} (Min:{self.min_face_size}px, Dim:{self.emb_dim})")
    def _init_deepface(self):
        try:
            res=DeepFace.represent(np.zeros((max(64,self.min_face_size),max(64,self.min_face_size),3),dtype=np.uint8), model_name=self.model_name, enforce_detection=False, detector_backend=self.detector_backend, align=False)
            self.emb_dim = len(res[0]["embedding"]) if isinstance(res,list) and res and "embedding" in res[0] else {'Facenet512':512, 'VGG-Face':2622}.get(self.model_name,512)
        except Exception as e: logger.error(f"DeepFace modeli '{self.model_name}' yüklenemedi: {e}"); raise RuntimeError(f"DeepFace yüklenemedi: {e}")
    
    def get_embedding_from_pil(self, image_pil: Image.Image) -> Optional[np.ndarray]:
        """Verilen bir PIL Image'dan (RGB formatında olduğu varsayılır) embedding çıkarır."""
        try:
            # Boyut kontrolü zaten _get_image_paths_for_faiss içinde yapılıyor,
            # ama burada da bir güvenlik katmanı olarak kalabilir veya extractor'a güvenilebilir.
            if image_pil.width < self.min_face_size or image_pil.height < self.min_face_size: 
                logger.debug(f"Görüntü ({image_pil.size}) embedding için çok küçük, atlanıyor.")
                return None
            
            img_np_rgb = np.array(image_pil) # PIL Image'dan direkt numpy array
            representation = DeepFace.represent(
                img_path=img_np_rgb, 
                model_name=self.model_name,
                enforce_detection=False, 
                detector_backend=self.detector_backend,
                align=False
            )
            if isinstance(representation, list) and representation and "embedding" in representation[0]:
                embedding_vector = np.array(representation[0]["embedding"])
                if embedding_vector.shape[0] == self.emb_dim:
                    return embedding_vector
                else:
                    logger.warning(f"Beklenmedik embedding boyutu ({embedding_vector.shape[0]}). Beklenen: {self.emb_dim}")
            else:
                 logger.warning(f"Geçerli embedding formatı alınamadı. Dönen: {type(representation)}")
            return None
        except UnidentifiedImageError: 
            logger.error(f"Geçersiz veya bozuk görüntü dosyası (PIL).") # Hangi dosya olduğunu bilmek iyi olurdu, çağıran loglamalı
            return None
        except Exception as e: 
            logger.error(f"PIL görüntüden embedding çıkarılırken hata: {e}")
            return None

class FaissDBHandler: # Öncekiyle aynı
    def __init__(self, emb_dim: int): self.emb_dim = emb_dim
    def create_and_save_index(self, embs:np.ndarray, ids:List[str], idx_path:Path, map_path:Path) -> bool:
        if embs.ndim == 1 and embs.shape[0] == 0 : embs = embs.reshape(0, self.emb_dim)
        if embs.shape[0]!=len(ids) or (embs.shape[0]>0 and embs.shape[1]!=self.emb_dim):
            logger.error(f"Embedding/ID sayısı ({embs.shape[0]},{len(ids)}) veya boyut ({embs.shape[1] if embs.ndim > 1 and embs.shape[0]>0 else 'N/A'}) uyumsuz."); return False
        try:
            logger.info(f"Yeni FAISS '{idx_path.name}' ({embs.shape[0]} vektör) oluşturuluyor...")
            index=faiss.IndexFlatIP(self.emb_dim)
            if embs.shape[0] > 0: index.add(np.array([normalize_vector(e.astype(np.float32)) for e in embs]))
            faiss.write_index(index,str(idx_path));
            with open(map_path,'w',encoding='utf-8') as f: json.dump(ids,f,indent=4,ensure_ascii=False)
            logger.info(f"Yeni FAISS ve harita kaydedildi: '{idx_path.name}', '{map_path.name}'")
            return True
        except Exception as e: logger.error(f"Yeni FAISS oluşturma/kaydetme hatası: {e}",exc_info=True); return False

# Augmentasyon fonksiyonu artık kullanılmayacak, isteğe bağlı olarak silinebilir.
# def augment_pil_image(image_pil: Image.Image) -> List[Image.Image]:
#     # ...

# --- Ana FAISS Oluşturma Sınıfı ---
class FaissBuilder:
    def __init__(self, config: FaissStarterConfig):
        self.config = config
        self.extractor = EmbeddingExtractor(
            config.EMBEDDING_MODEL_NAME,
            config.DEEPFACE_DETECTOR_BACKEND_FOR_CROPS,
            config.MIN_FACE_SIZE_FOR_PROCESSING
        )
        if self.extractor.emb_dim == -1:
            raise RuntimeError("FaissBuilder için EmbeddingExtractor başlatılamadı.")
        self.faiss_handler = FaissDBHandler(self.extractor.emb_dim)
        self.stats = defaultdict(int)

    def _get_class_image_paths_for_faiss(self, class_id_str: str) -> List[Path]: # Öncekiyle aynı (rglob'lu versiyon)
        class_path = self.config.CLASSES_DIR / class_id_str
        image_paths: List[Path] = []
        logger.info(f"'{class_id_str}' sınıfı için '{class_path}' ve tüm alt klasörleri taranıyor...")
        if not class_path.exists() or not class_path.is_dir():
            logger.warning(f"Sınıf klasörü bulunamadı veya bir dizin değil: {class_path}"); return image_paths
        supported_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff']
        for item_path in class_path.rglob('*'):
            if item_path.is_file() and item_path.suffix.lower() in supported_extensions:
                try:
                    with Image.open(item_path) as img_check: 
                        if img_check.width >= self.config.MIN_FACE_SIZE_FOR_PROCESSING and \
                           img_check.height >= self.config.MIN_FACE_SIZE_FOR_PROCESSING:
                            image_paths.append(item_path)
                            logger.debug(f"FAISS için eklendi: {item_path} (Boyut: {img_check.size})")
                        else:
                            logger.debug(f"FAISS için '{item_path.name}' ({img_check.size}) boyutu yetersiz, atlanıyor.")
                            self.stats[f'img_skipped_too_small_in_{class_id_str}'] += 1
                except UnidentifiedImageError:
                    logger.warning(f"FAISS için '{item_path.name}' bozuk/tanımlanamayan resim, atlanıyor.")
                    self.stats[f'img_skipped_corrupt_in_{class_id_str}'] += 1
                except Exception as e_open:
                    logger.error(f"'{item_path}' resmi FAISS için açılırken/kontrol edilirken hata: {e_open}")
                    self.stats[f'img_skipped_error_in_{class_id_str}'] += 1
        if not image_paths: logger.warning(f"FAISS için '{class_id_str}' sınıfında (ve alt k.) hiç uygun resim bulunamadı.")
        else: logger.info(f"FAISS için '{class_id_str}' sınıfından (ve alt k.) toplam {len(image_paths)} resim bulundu.")
        return image_paths

    def build_production_faiss_with_prototypes(self):
        logger.info("--- Production FAISS İndeksi Oluşturuluyor (Sınıf Prototip/Ortalamaları ile, AUGMENTASYONSUZ) ---")
        self.stats.clear()

        prototype_class_ids: List[str] = []
        prototype_embeddings: List[np.ndarray] = []

        if not self.config.CLASSES_DIR.exists() or not any(self.config.CLASSES_DIR.iterdir()):
            logger.warning(f"Kaynak sınıf klasörü '{self.config.CLASSES_DIR}' boş veya bulunamadı. Boş FAISS oluşturulacak.")
            self.faiss_handler.create_and_save_index(np.array([]).reshape(0,self.extractor.emb_dim), [], self.config.PRODUCTION_FAISS_INDEX_PATH, self.config.PRODUCTION_FAISS_MAP_PATH)
            return

        class_folders_to_process = [d for d in self.config.CLASSES_DIR.iterdir() if d.is_dir()]
        logger.info(f"Production FAISS (prototip) için {len(class_folders_to_process)} sınıf klasörü bulundu.")

        for class_dir in class_folders_to_process:
            class_id = class_dir.name
            image_paths_for_prototype = self._get_class_image_paths_for_faiss(class_id)
            
            if not image_paths_for_prototype:
                logger.warning(f"'{class_id}' sınıfı için prototip oluşturacak resim bulunamadı, atlanıyor.")
                self.stats['classes_skipped_for_prototype_no_images'] += 1
                continue

            self.stats['classes_processed_for_prototype'] += 1
            
            class_specific_embeddings_for_avg: List[np.ndarray] = []
            images_embedded_count = 0
            for img_path in image_paths_for_prototype:
                try:
                    img_pil = Image.open(img_path).convert("RGB") # RGB olduğundan emin ol
                    # Augmentasyon YAPILMIYOR
                    embedding_vector = self.extractor.get_embedding_from_pil(img_pil)

                    if embedding_vector is not None:
                        class_specific_embeddings_for_avg.append(embedding_vector)
                        images_embedded_count += 1
                    # else: extractor zaten logladı
                except Exception as e:
                    logger.error(f"'{img_path.name}' prototip için embedding çıkarılırken hata: {e}")
            
            if class_specific_embeddings_for_avg:
                calculated_prototype_vector = np.mean(np.array(class_specific_embeddings_for_avg), axis=0)
                prototype_embeddings.append(calculated_prototype_vector)
                prototype_class_ids.append(class_id)
                logger.info(f"'{class_id}' sınıfı için {images_embedded_count} resimden prototip oluşturuldu ve FAISS listesine eklendi.")
                self.stats[f'prototype_images_used_for_{class_id}'] = images_embedded_count
            else:
                logger.warning(f"'{class_id}' sınıfı için HİÇ geçerli embedding bulunamadı, prototip oluşturulamadı.")
                self.stats['prototype_creation_failed_no_embeddings'] +=1
        
        if not prototype_embeddings:
            logger.error("Nihai Production FAISS oluşturmak için HİÇ prototip embedding toplanamadı!")
            self.faiss_handler.create_and_save_index(np.array([]).reshape(0,self.extractor.emb_dim), [], self.config.PRODUCTION_FAISS_INDEX_PATH, self.config.PRODUCTION_FAISS_MAP_PATH)
        else:
            prototype_embeddings_np = np.array(prototype_embeddings)
            ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            pidx_path = self.config.PRODUCTION_FAISS_INDEX_PATH
            pmap_path = self.config.PRODUCTION_FAISS_MAP_PATH
            if pidx_path.exists(): shutil.move(str(pidx_path), str(pidx_path.parent / f"{pidx_path.stem}_backup_{ts}{pidx_path.suffix}"))
            if pmap_path.exists(): shutil.move(str(pmap_path), str(pmap_path.parent / f"{pmap_path.stem}_backup_{ts}{pmap_path.suffix}"))
            if self.faiss_handler.create_and_save_index(prototype_embeddings_np, prototype_class_ids, pidx_path, pmap_path):
                logger.info("Yeni Production FAISS indeksi (prototiplerle) başarıyla oluşturuldu.")
                self.stats['production_faiss_total_prototypes'] = prototype_embeddings_np.shape[0]
                self.stats['production_faiss_total_classes'] = len(set(prototype_class_ids))
            else: logger.error("Yeni Production FAISS indeksi (prototiplerle) oluşturulamadı!")
        logger.info("--- Production FAISS İndeksi Oluşturma (Prototip Tabanlı) Tamamlandı ---"); logger.info("İstatistikler:")
        for key, value in sorted(self.stats.items()): logger.info(f"  {key}: {value}")

if __name__ == "__main__":
    cfg = FaissStarterConfig()
    logger.info(f"Faiss Starter (Prototip Tabanlı, Augmentasyonsuz) başlatılıyor. Kaynak Sınıf Klasörü: {cfg.CLASSES_DIR}")
    logger.info(f"Hedef Production FAISS Index: {cfg.PRODUCTION_FAISS_INDEX_PATH}")
    logger.info(f"Hedef Production FAISS Map: {cfg.PRODUCTION_FAISS_MAP_PATH}")

    if not cfg.CLASSES_DIR.exists():
        logger.critical(f"Kaynak sınıf klasörü '{cfg.CLASSES_DIR}' bulunamadı! Script sonlandırılıyor.")
    elif not any(item.is_dir() for item in cfg.CLASSES_DIR.iterdir() if item.name not in cfg.HUMANS_DIR.glob("*") ): # Basit bir kontrol, daha iyi yapılabilir
        logger.warning(f"'{cfg.CLASSES_DIR}' içinde işlenecek sınıf alt klasörü bulunamadı. Boş bir FAISS indeksi oluşturulabilir.")
    
    try:
        builder = FaissBuilder(cfg)
        builder.build_production_faiss_with_prototypes()
    except RuntimeError as e: 
        logger.critical(f"Kritik başlatma hatası: {e}")
    except Exception as e:
        logger.critical(f"Beklenmedik genel bir hata oluştu: {e}", exc_info=True)