import datetime
import json
import logging
import os
import shutil
import joblib # scikit-learn modellerini kaydetmek/yüklemek için
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Set
from collections import defaultdict

import numpy as np
from PIL import Image, UnidentifiedImageError, ImageOps
from deepface import DeepFace
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mlp_face_classifier.log", mode='a')
    ]
)
logger = logging.getLogger(__name__)

# --- Yapılandırma Sınıfı (face_data_manager.py'den uyarlanmış) ---
class ClassifierConfig:
    BASE_DB_DIR = Path("/home/earsal@ETE.local/Desktop/codes/seeing-any/project-02/yenidenemem_few_shot/face_teslim/face_db")
    HUMANS_DIR_NAME = "Human"
    CLASSES_SUBDIR_NAME = "Classes" # Eğitim için sınıfların okunacağı yer
    
    # Kaydedilecek model ve encoder için yer
    SAVED_MODELS_DIR_NAME = "/home/earsal@ETE.local/Desktop/codes/seeing-any/project-02/yenidenemem_few_shot/face_teslim/model_deepface_yolo/saved_models"
    MODEL_FILENAME = "mlp_classifier.joblib"
    ENCODER_FILENAME = "label_encoder.joblib"

    # Embedding için ayarlar (face_data_manager.py ile tutarlı olmalı)
    EMBEDDING_MODEL_NAME = 'Facenet512'
    DEEPFACE_DETECTOR_BACKEND_FOR_CROPS = 'skip'
    MIN_FACE_SIZE_FOR_PROCESSING = 30
    
    # Eğitim için
    APPLY_AUGMENTATION_FOR_TRAINING = True # Eğitim verisini çoğalt
    TEST_SPLIT_SIZE = 0.2 # Verinin ne kadarının test için ayrılacağı
    RANDOM_STATE_SEED = 42 # Tekrarlanabilirlik için

    # MLP Model Parametreleri (Bunlar ayarlanabilir)
    MLP_HIDDEN_LAYER_SIZES = (256, 128) # Örnek: İki gizli katman
    MLP_ACTIVATION = 'relu'
    MLP_SOLVER = 'adam'
    MLP_MAX_ITER = 500 # Eğitim iterasyon sayısı
    MLP_LEARNING_RATE_INIT = 0.001
    MLP_EARLY_STOPPING = True
    MLP_N_ITER_NO_CHANGE = 20 # Erken durdurma için sabır

    # Alt klasör isimleri (face_data_manager.py ile tutarlı)
    FROM_DETECTED_SUBDIR_NAME = "from_detected"
    IMAGES_SUBDIR_FOR_NUMERIC_CLASSES = "images"
    REFERENCE_IMAGES_SUBDIR_FOR_NAMED = "reference_images"


    def __init__(self):
        self.HUMANS_DIR = self.BASE_DB_DIR / self.HUMANS_DIR_NAME
        self.CLASSES_DIR = self.HUMANS_DIR / self.CLASSES_SUBDIR_NAME
        
        self.SAVED_MODELS_PATH = self.HUMANS_DIR / self.SAVED_MODELS_DIR_NAME
        self.SAVED_MODELS_PATH.mkdir(parents=True, exist_ok=True)
        self.MODEL_SAVE_PATH = self.SAVED_MODELS_PATH / self.MODEL_FILENAME
        self.ENCODER_SAVE_PATH = self.SAVED_MODELS_PATH / self.ENCODER_FILENAME

# --- Embedding Çıkarıcı (face_data_manager.py'den uyarlanmış) ---
class EmbeddingExtractor:
    def __init__(self, model_name: str, detector_backend: str, min_face_size: int):
        self.model_name,self.detector_backend,self.min_face_size=model_name,detector_backend,min_face_size
        self.emb_dim:int=-1; self._init_deepface()
        logger.info(f"MLP-Extractor: {self.model_name} (Min:{self.min_face_size}px, Dim:{self.emb_dim})")
    def _init_deepface(self):
        try:
            res=DeepFace.represent(np.zeros((64,64,3),dtype=np.uint8), model_name=self.model_name, enforce_detection=False, detector_backend=self.detector_backend, align=False)
            self.emb_dim = len(res[0]["embedding"]) if isinstance(res,list) and res and "embedding" in res[0] else {'Facenet512':512}.get(self.model_name,512)
        except Exception as e: logger.error(f"MLP-DeepFace modeli '{self.model_name}' yüklenemedi: {e}"); raise RuntimeError(f"MLP-DeepFace yüklenemedi: {e}")
    def get_embedding_from_pil(self, image_pil: Image.Image) -> Optional[np.ndarray]: # PIL Image alır
        try:
            if image_pil.width < self.min_face_size or image_pil.height < self.min_face_size: return None
            res = DeepFace.represent(np.array(image_pil), model_name=self.model_name, enforce_detection=False, detector_backend=self.detector_backend, align=False)
            if isinstance(res, list) and res and "embedding" in res[0] and len(res[0]["embedding"]) == self.emb_dim:
                return np.array(res[0]["embedding"])
            return None
        except Exception as e:
            logger.error(f"PIL görüntüden embedding çıkarılırken hata: {e}")
            return None

# --- Augmentasyon Fonksiyonu ---
def augment_pil_image(image_pil: Image.Image) -> List[Image.Image]:
    augmented_images = [image_pil] # Orijinali her zaman ekle
    augmented_images.append(ImageOps.mirror(image_pil))
    # İsteğe bağlı olarak daha fazla augmentasyon eklenebilir (parlaklık, kontrast vb.)
    return augmented_images

# --- Veri Yükleme ve Hazırlama ---
def load_and_prepare_training_data(config: ClassifierConfig, extractor: EmbeddingExtractor) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[LabelEncoder]]:
    logger.info("Eğitim verisi yükleniyor ve hazırlanıyor...")
    all_embeddings: List[np.ndarray] = []
    all_labels: List[str] = []

    if not config.CLASSES_DIR.exists():
        logger.error(f"Sınıf klasörü bulunamadı: {config.CLASSES_DIR}")
        return None, None, None

    class_folders = [d for d in config.CLASSES_DIR.iterdir() if d.is_dir()]
    if not class_folders:
        logger.error(f"{config.CLASSES_DIR} altında hiç sınıf klasörü bulunamadı.")
        return None, None, None
    
    logger.info(f"{len(class_folders)} sınıf klasörü bulundu: {[cf.name for cf in class_folders]}")

    for class_dir in class_folders:
        class_id = class_dir.name
        logger.info(f"'{class_id}' sınıfı için resimler işleniyor...")
        
        image_paths_for_class: List[Path] = []
        # Resim toplama mantığı (face_data_manager.py'deki _get_class_image_paths'e benzer)
        source_subdirs = [class_dir] 
        if not class_id.isdigit(): # İsimli sınıflar
            source_subdirs.append(class_dir / config.REFERENCE_IMAGES_SUBDIR_FOR_NAMED)
            source_subdirs.append(class_dir / config.FROM_DETECTED_SUBDIR_NAME)
        else: # Sayısal ID'li sınıflar
             source_subdirs.append(class_dir / config.IMAGES_SUBDIR_FOR_NUMERIC_CLASSES)
        
        processed_files_in_class: Set[Path] = set()
        images_found_for_this_class = 0
        for subdir in source_subdirs:
            if subdir.exists() and subdir.is_dir():
                for item in subdir.iterdir():
                    if item.is_file() and item.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        resolved_item = item.resolve()
                        if resolved_item not in processed_files_in_class:
                            try:
                                img_pil = Image.open(item).convert("RGB")
                                images_to_process = augment_pil_image(img_pil) if config.APPLY_AUGMENTATION_FOR_TRAINING else [img_pil]
                                
                                for aug_img_pil in images_to_process:
                                    emb = extractor.get_embedding_from_pil(aug_img_pil)
                                    if emb is not None:
                                        all_embeddings.append(emb)
                                        all_labels.append(class_id)
                                        images_found_for_this_class +=1
                                processed_files_in_class.add(resolved_item)
                            except UnidentifiedImageError:
                                logger.warning(f"Bozuk resim: {item}")
                            except Exception as e_img:
                                logger.error(f"'{item}' işlenirken hata: {e_img}")
        logger.info(f"'{class_id}' sınıfından {images_found_for_this_class} embedding (augmentasyon dahil) eklendi.")

    if not all_embeddings or not all_labels:
        logger.error("Eğitim için hiç geçerli embedding veya etiket bulunamadı.")
        return None, None, None

    X = np.array(all_embeddings)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(np.array(all_labels))
    
    logger.info(f"Toplam {X.shape[0]} embedding, {len(label_encoder.classes_)} farklı sınıftan yüklendi.")
    logger.info(f"Sınıflar: {label_encoder.classes_}")
    
    return X, y_encoded, label_encoder

# --- Model Eğitimi ---
def train_mlp_classifier(config: ClassifierConfig, X: np.ndarray, y_encoded: np.ndarray, label_encoder: LabelEncoder):
    logger.info("MLP sınıflandırıcı modeli eğitiliyor...")
    
    # Veriyi eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=config.TEST_SPLIT_SIZE, 
        random_state=config.RANDOM_STATE_SEED,
        stratify=y_encoded # Sınıf dağılımını koru
    )
    logger.info(f"Eğitim seti boyutu: {X_train.shape[0]}, Test seti boyutu: {X_test.shape[0]}")

    model = MLPClassifier(
        hidden_layer_sizes=config.MLP_HIDDEN_LAYER_SIZES,
        activation=config.MLP_ACTIVATION,
        solver=config.MLP_SOLVER,
        max_iter=config.MLP_MAX_ITER,
        learning_rate_init=config.MLP_LEARNING_RATE_INIT,
        early_stopping=config.MLP_EARLY_STOPPING,
        n_iter_no_change=config.MLP_N_ITER_NO_CHANGE,
        random_state=config.RANDOM_STATE_SEED,
        verbose=True # Eğitim sürecini logla
    )

    try:
        model.fit(X_train, y_train)
    except ValueError as ve:
        logger.error(f"Model eğitimi sırasında ValueError: {ve}")
        logger.error("Bu genellikle bir veya daha fazla sınıfın test setinde yeterli örneği olmamasından kaynaklanabilir (stratify'a rağmen).")
        logger.error("Sınıf başına düşen örnek sayısını kontrol edin veya test_size'ı azaltmayı deneyin.")
        return

    logger.info("Model eğitimi tamamlandı.")

    # Modeli test et
    y_pred_test = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    logger.info(f"Test Seti Doğruluğu (Accuracy): {accuracy:.4f}")
    
    logger.info("Test Seti Sınıflandırma Raporu:")
    # Etiketleri orijinal isimlerine çevirerek raporu daha okunabilir yap
    try:
        y_test_original_labels = label_encoder.inverse_transform(y_test)
        y_pred_original_labels = label_encoder.inverse_transform(y_pred_test)
        report = classification_report(y_test_original_labels, y_pred_original_labels, zero_division=0)
        logger.info("\n" + report)
    except Exception as e_report:
        logger.error(f"Sınıflandırma raporu oluşturulurken hata: {e_report}")
        logger.info("Sayısal etiketlerle rapor:")
        report_numeric = classification_report(y_test, y_pred_test, zero_division=0)
        logger.info("\n" + report_numeric)


    # Modeli ve LabelEncoder'ı kaydet
    try:
        joblib.dump(model, config.MODEL_SAVE_PATH)
        logger.info(f"Eğitilmiş model şuraya kaydedildi: {config.MODEL_SAVE_PATH}")
        joblib.dump(label_encoder, config.ENCODER_SAVE_PATH)
        logger.info(f"Label encoder şuraya kaydedildi: {config.ENCODER_SAVE_PATH}")
    except Exception as e_save:
        logger.error(f"Model veya encoder kaydedilirken hata: {e_save}")

# --- Model Yükleme ve Tahmin ---
class TrainedMLPClassifier:
    def __init__(self, model_path: Path, encoder_path: Path):
        self.model: Optional[MLPClassifier] = None
        self.label_encoder: Optional[LabelEncoder] = None
        if model_path.exists() and encoder_path.exists():
            try:
                self.model = joblib.load(model_path)
                self.label_encoder = joblib.load(encoder_path)
                logger.info(f"MLP modeli '{model_path}' ve encoder '{encoder_path}' başarıyla yüklendi.")
                logger.info(f"Modelin tanıdığı sınıflar: {list(self.label_encoder.classes_)}")
            except Exception as e:
                logger.error(f"Kaydedilmiş model veya encoder yüklenirken hata: {e}")
                self.model = None
                self.label_encoder = None
        else:
            logger.error(f"Model ({model_path}) veya encoder ({encoder_path}) bulunamadı.")

    def is_ready(self) -> bool:
        return self.model is not None and self.label_encoder is not None

    def predict(self, embedding: np.ndarray) -> Optional[Tuple[str, float]]:
        if not self.is_ready() or self.model is None or self.label_encoder is None: # type checker için
            logger.error("Model veya encoder yüklenmediği için tahmin yapılamıyor.")
            return None
        
        try:
            embedding_reshaped = embedding.reshape(1, -1) # Tek örnek için (1, N_features)
            probabilities = self.model.predict_proba(embedding_reshaped)[0] # İlk (ve tek) örnek için olasılıklar
            predicted_encoded_label = np.argmax(probabilities)
            confidence = probabilities[predicted_encoded_label]
            predicted_original_label = self.label_encoder.inverse_transform([predicted_encoded_label])[0]
            return predicted_original_label, float(confidence)
        except Exception as e:
            logger.error(f"Embedding için tahmin yapılırken hata: {e}")
            return None

# --- Ana Çalıştırma Bloğu ---
if __name__ == "__main__":
    cfg = ClassifierConfig()
    extractor = EmbeddingExtractor(
        cfg.EMBEDDING_MODEL_NAME, 
        cfg.DEEPFACE_DETECTOR_BACKEND_FOR_CROPS, 
        cfg.MIN_FACE_SIZE_FOR_PROCESSING
    )

    # Basit bir argüman ayrıştırma (eğitim mi tahmin mi)
    import argparse
    parser = argparse.ArgumentParser(description="MLP Yüz Sınıflandırıcı Eğitim ve Tahmin Scripti")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "predict_sample"], 
                        help="'train' veya 'predict_sample'")
    parser.add_argument("--image_path", type=str, help="Tahmin için örnek resim yolu (predict_sample modu için)")
    args = parser.parse_args()

    if args.mode == "train":
        logger.info("--- EĞİTİM MODU ---")
        X_data, y_data_encoded, encoder = load_and_prepare_training_data(cfg, extractor)
        if X_data is not None and y_data_encoded is not None and encoder is not None:
            if len(encoder.classes_) < 2:
                logger.error(f"Eğitim için en az 2 farklı sınıf gereklidir. Bulunan: {len(encoder.classes_)}. Lütfen 'humans/classes/' klasörünüzü kontrol edin.")
            else:
                train_mlp_classifier(cfg, X_data, y_data_encoded, encoder)
        else:
            logger.error("Veri yükleme başarısız olduğu için eğitime devam edilemiyor.")

    elif args.mode == "predict_sample":
        logger.info("--- ÖRNEK TAHMİN MODU ---")
        if not args.image_path:
            logger.error("Tahmin için --image_path ile bir resim yolu belirtmelisiniz.")
        else:
            sample_image_path = Path(args.image_path)
            if not sample_image_path.exists():
                logger.error(f"Belirtilen resim yolu bulunamadı: {sample_image_path}")
            else:
                classifier = TrainedMLPClassifier(cfg.MODEL_SAVE_PATH, cfg.ENCODER_SAVE_PATH)
                if classifier.is_ready():
                    try:
                        img_pil_sample = Image.open(sample_image_path).convert("RGB")
                        sample_embedding = extractor.get_embedding_from_pil(img_pil_sample)
                        if sample_embedding is not None:
                            prediction = classifier.predict(sample_embedding)
                            if prediction:
                                label, conf = prediction
                                logger.info(f"Tahmin -> Resim: {sample_image_path.name}, Etiket: {label}, Güven: {conf:.4f}")
                            else:
                                logger.error("Örnek resim için tahmin alınamadı.")
                        else:
                            logger.error(f"'{sample_image_path.name}' için embedding çıkarılamadı.")
                    except Exception as e_pred_sample:
                        logger.error(f"Örnek tahmin sırasında hata: {e_pred_sample}")
                else:
                    logger.error("Model yüklenemediği için örnek tahmin yapılamıyor. Lütfen önce modeli eğitin.")
    else:
        logger.error(f"Geçersiz mod: {args.mode}. 'train' veya 'predict_sample' kullanın.")