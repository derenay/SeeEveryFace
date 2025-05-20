import datetime
import json
import os
import time 
from pathlib import Path
import cv2
import faiss
import numpy as np
import pandas as pd 
from deepface import DeepFace
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

EMBEDDING_MODEL_NAME = 'Facenet512'
YOLO_FACE_DETECTION_MODEL = 'project-02/yolov11m-face.pt'
DEEPFACE_DETECTOR_BACKEND_FOR_CROPS = 'skip' 
DEFAULT_DISTANCE_THRESHOLD = 0.50 
FAISS_INDEX_FILE = 'face_index_v5_facenet512.index'
INDEX_MAP_FILE = 'face_index_map_v5.json'
RESULTS_BASE_DIR = "results_face" 

try:
    FONT_PATH = "arial.ttf" if os.name == 'nt' else "DejaVuSans.ttf"
    ImageFont.truetype(FONT_PATH, 15)
except IOError:
    FONT_PATH = None
    print(f"Uyarı: Belirtilen font ('{FONT_PATH if FONT_PATH else 'arial.ttf/DejaVuSans.ttf'}') bulunamadı. Varsayılan font kullanılacak.")

yolo_model = None
deepface_ready = False
faiss_index = None
index_to_name_map = []
embedding_dimension = -1 

def initialize_models():
    """Loads YOLO and DeepFace models, determines embedding dimension."""
    global yolo_model, deepface_ready, embedding_dimension
    print("Modeller yükleniyor...")
    if yolo_model is None:
        if not os.path.exists(YOLO_FACE_DETECTION_MODEL):
            print(f"HATA: YOLO modeli '{YOLO_FACE_DETECTION_MODEL}' bulunamadı.")
            raise FileNotFoundError(f"YOLO model file not found: {YOLO_FACE_DETECTION_MODEL}")
        try:
            yolo_model = YOLO(YOLO_FACE_DETECTION_MODEL)
            print(f"YOLO modeli '{YOLO_FACE_DETECTION_MODEL}' başarıyla yüklendi.")
        except Exception as e:
            print(f"HATA: YOLO modeli yüklenemedi: {e}")
            raise RuntimeError(f"Failed to load YOLO model: {e}") 

    if not deepface_ready:
        try:
            dummy_img_for_warmup = np.zeros((64, 64, 3), dtype=np.uint8)
            embedding_result = DeepFace.represent(
                dummy_img_for_warmup,
                model_name=EMBEDDING_MODEL_NAME,
                enforce_detection=False,
                detector_backend=DEEPFACE_DETECTOR_BACKEND_FOR_CROPS
            )
            if isinstance(embedding_result, list) and len(embedding_result) > 0 and "embedding" in embedding_result[0]:
                embedding_dimension = len(embedding_result[0]["embedding"])
                print(f"DeepFace embedding modeli '{EMBEDDING_MODEL_NAME}' (Boyut: {embedding_dimension}) kullanıma hazır.")
                deepface_ready = True
            else:

                embedding_dimension = 512 
                print(f"UYARI: DeepFace.represent'ten embedding boyutu alınamadı. Varsayılan {embedding_dimension} kullanılıyor.")
                deepface_ready = True 
        except Exception as e:
            print(f"HATA: DeepFace modeli '{EMBEDDING_MODEL_NAME}' yüklenemedi/kullanılamadı: {e}")
            raise RuntimeError(f"Failed to load DeepFace model: {e}") 

    print("Modeller başarıyla yüklendi.")


def load_existing_database():
    """Loads the pre-built FAISS index and its name map."""
    global faiss_index, index_to_name_map, embedding_dimension
    print("Mevcut FAISS veritabanı yükleniyor...")

    if embedding_dimension <= 0:
        print("HATA: Embedding boyutu bilinmiyor. Önce modelleri başlatın.")
        raise RuntimeError("Embedding dimension unknown, cannot load database.")

    if not os.path.exists(INDEX_MAP_FILE):
        print(f"HATA: FAISS index harita dosyası '{INDEX_MAP_FILE}' bulunamadı.")
        raise FileNotFoundError(f"FAISS index map file not found: {INDEX_MAP_FILE}")
    try:
        with open(INDEX_MAP_FILE, 'r', encoding='utf-8') as f:
            index_to_name_map = json.load(f)
        print(f"FAISS index haritası '{INDEX_MAP_FILE}' yüklendi ({len(index_to_name_map)} girdi).")
    except Exception as e:
        print(f"HATA: FAISS index haritası yüklenirken: {e}")
        raise RuntimeError(f"Failed to load FAISS index map: {e}")

    if not index_to_name_map:
        print("HATA: FAISS index haritası boş. Tanıma yapılamaz.")
        raise ValueError("FAISS index map is empty.")

    if not os.path.exists(FAISS_INDEX_FILE):
        print(f"HATA: FAISS index dosyası '{FAISS_INDEX_FILE}' bulunamadı.")
        raise FileNotFoundError(f"FAISS index file not found: {FAISS_INDEX_FILE}")
    try:
        faiss_index = faiss.read_index(FAISS_INDEX_FILE)
        print(f"FAISS index '{FAISS_INDEX_FILE}' yüklendi ({faiss_index.ntotal} vektör).")

        if faiss_index.ntotal != len(index_to_name_map):
            msg = f"HATA: FAISS index boyutu ({faiss_index.ntotal}) ile index haritası boyutu ({len(index_to_name_map)}) eşleşmiyor!"
            print(msg)
            raise ValueError(msg)

        if faiss_index.d != embedding_dimension:
             msg = f"HATA: Yüklenen FAISS index boyutu ({faiss_index.d}) ile beklenen embedding boyutu ({embedding_dimension}) farklı!"
             print(msg)
             raise ValueError(msg)

    except Exception as e:
        print(f"HATA: FAISS index yüklenirken: {e}")
        raise RuntimeError(f"Failed to load FAISS index: {e}")

    print("FAISS veritabanı başarıyla yüklendi.")


def load_font(size=15):
    """Loads the preferred font or falls back to default."""
    if FONT_PATH:
        try:
            return ImageFont.truetype(FONT_PATH, size)
        except IOError: pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def normalize_vector(vec):
    """Normalizes a numpy vector for cosine similarity using inner product."""
    norm = np.linalg.norm(vec)
    if norm < 1e-6: 

       return np.zeros_like(vec).astype('float32')
    return (vec / norm).astype('float32')

def get_face_crop_from_image_pil(image_pil, yolo_detector_instance):
    """Detects faces using YOLO, returns the largest face crop PIL image."""
    if yolo_detector_instance is None: return None
    try:
        results = yolo_detector_instance.predict(image_pil, verbose=False, conf=0.3)
    except Exception as e:
        print(f"YOLO ile yüz tespiti sırasında hata: {e}")
        return None

    best_box = None
    max_area = 0
    if results:
        for result in results:
            if result.boxes:
                for box_data in result.boxes:
                    conf = box_data.conf[0] if box_data.conf is not None and len(box_data.conf) > 0 else 0
                    if conf > 0.3:
                        if box_data.xyxy is not None and len(box_data.xyxy) > 0:
                            x1, y1, x2, y2 = map(int, box_data.xyxy[0].tolist())
                            if x1 < x2 and y1 < y2:
                                area = (x2 - x1) * (y2 - y1)
                                if area > max_area:
                                    max_area = area
                                    best_box = (x1, y1, x2, y2)


    if best_box:
        img_width, img_height = image_pil.size
        x1_c = max(0, best_box[0])
        y1_c = max(0, best_box[1])
        x2_c = min(img_width, best_box[2])
        y2_c = min(img_height, best_box[3])
        if x1_c < x2_c and y1_c < y2_c:
            return image_pil.crop((x1_c, y1_c, x2_c, y2_c))
        else:
             return None
    return None

def get_embedding_from_face_pil(face_crop_pil):
    """Extracts embedding from a PIL face crop using DeepFace."""
    global deepface_ready, embedding_dimension
    if not deepface_ready or face_crop_pil is None: return None
    if face_crop_pil.width < 20 or face_crop_pil.height < 20: return None

    try:
        face_crop_numpy_rgb = np.array(face_crop_pil.convert('RGB'))
        face_crop_numpy_bgr = cv2.cvtColor(face_crop_numpy_rgb, cv2.COLOR_RGB2BGR)
        embedding_objs = DeepFace.represent(
            img_path=face_crop_numpy_bgr, model_name=EMBEDDING_MODEL_NAME,
            enforce_detection=False, detector_backend=DEEPFACE_DETECTOR_BACKEND_FOR_CROPS
        )
        if embedding_objs and isinstance(embedding_objs, list) and len(embedding_objs) > 0:
            embedding = np.array(embedding_objs[0]["embedding"])
            if embedding_dimension > 0 and embedding.shape[0] != embedding_dimension:
                 print(f"HATA: Embedding boyutu ({embedding.shape[0]}) != beklenen ({embedding_dimension}).")
                 return None
            return embedding
        return None
    except ValueError as ve:
        if "shape" in str(ve).lower():
            print(f"Embedding çıkarılırken boyut uyumsuzluğu hatası: {ve}. Yüz: {face_crop_pil.size}")
        elif "could not be detected" in str(ve).lower():
             pass 
        else:
             print(f"Embedding çıkarılırken ValueError: {ve}")
        return None
    except Exception as e:
        err_str = str(e).lower()
        if "face could not be detected" not in err_str and "singleton array" not in err_str:
             print(f"Embedding çıkarılırken beklenmedik hata: {e}")
        return None



def recognize_faces_in_frame(frame_pil, threshold):
    """
    Detects and recognizes faces in a single PIL frame using FAISS.

    Args:
        frame_pil (PIL.Image.Image): The input image frame.
        threshold (float): The cosine distance threshold for recognition.

    Returns:
        tuple:
            - PIL.Image.Image: The frame annotated with bounding boxes and labels.
            - list: A list of dictionaries, each containing details of a detected face.
                    Example: {'box': [x1,y1,x2,y2], 'identity': str, 'distance': float,
                              'status': str, 'closest_match': str|None}
    """
    global yolo_model, faiss_index, index_to_name_map

    annotated_frame = frame_pil.copy()
    draw = ImageDraw.Draw(annotated_frame)
    font_size = max(12, int(frame_pil.height / 45))
    font = load_font(size=font_size)
    recognition_results = [] 

    if yolo_model is None: return frame_pil, []
    try:
        detections = yolo_model.predict(frame_pil, verbose=False, conf=0.4)
    except Exception as e:
        print(f"YOLO predict hatası: {e}")
        return annotated_frame, [] 

    if not detections or not isinstance(detections, list):
        return annotated_frame, []

    for result_set in detections:
        if not hasattr(result_set, 'boxes') or result_set.boxes is None: continue

        for box_data in result_set.boxes:
            if box_data.xyxy is None or len(box_data.xyxy) == 0: continue
            if box_data.conf is None or len(box_data.conf) == 0: continue

            confidence = float(box_data.conf[0])
            if confidence < 0.4: continue 

            x1, y1, x2, y2 = map(int, box_data.xyxy[0].tolist())
            if x1 >= x2 or y1 >= y2: continue 

            face_crop_pil = frame_pil.crop((x1, y1, x2, y2))
            if face_crop_pil.width < 20 or face_crop_pil.height < 20: continue

          
            query_embedding = get_embedding_from_face_pil(face_crop_pil)

            identity = "Bilinmeyen"
            status = "NoEmbedding" 
            min_distance = float('inf')
            color = "red"
            best_match_name = None 

            if query_embedding is not None:
                status = "Unknown" 
                if faiss_index is not None and faiss_index.ntotal > 0:
                    try:
                        normalized_query = normalize_vector(query_embedding)
                        query_vector_2d = np.array([normalized_query]).astype('float32')

                        D, I = faiss_index.search(query_vector_2d, k=1)
                        if I.size > 0 and D.size > 0:
                            idx = I[0][0]
                            inner_product = D[0][0]
                    
                            similarity = max(0.0, min(1.0, inner_product))
                            min_distance = 1.0 - similarity # Cosine distance

                            if 0 <= idx < len(index_to_name_map):
                                best_match_name = index_to_name_map[idx]
                                if min_distance < threshold:
                                    identity = best_match_name
                                    color = "lime"
                                    status = "Recognized"
                                # else: status remains "Unknown"
                            else:
                                print(f"HATA: FAISS geçersiz index döndü: {idx}")
                                status = "Error" # Indicate an index mapping error
                        # else: status remains "Unknown" (search failed)

                    except Exception as faiss_err:
                        print(f"FAISS arama hatası: {faiss_err}")
                        status = "Error" #
                else:
                    status = "NoIndex" 
       
            result_data = {
                "box": [x1, y1, x2, y2],
                "identity": identity,
                "distance": round(min_distance, 4) if min_distance != float('inf') else None,
                "status": status,
                "closest_match": best_match_name 
            }
            recognition_results.append(result_data)

            # --- 7. Draw Annotation ---
            label_text = f"{identity}"
            if status == "Recognized":
                label_text += f" ({min_distance:.2f})"
            elif status == "Unknown" and best_match_name is not None:
                 label_text += f" ({best_match_name}?:{min_distance:.2f})" 
            elif status == "NoEmbedding":
                 label_text = "Ozellik Yok"
            elif status == "Error":
                 label_text = "Hata"

            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=max(1, int(frame_pil.height / 250)))
            try:
                text_bbox = draw.textbbox((x1, y1 - font_size - 2), label_text, font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
            except AttributeError: 
                try: text_w, text_h = draw.textsize(label_text, font=font)
                except AttributeError: text_w,text_h = len(label_text)*font_size*0.6,font_size 

            bg_y1 = y1 - text_h - 4
            if bg_y1 < 0: bg_y1 = y1 + 2
            draw.rectangle([(x1, bg_y1), (x1 + text_w + 4, bg_y1 + text_h + 2)], fill=color)
            draw.text((x1 + 2, bg_y1), label_text, fill="black", font=font)

    return annotated_frame, recognition_results


def process_image(image_path, output_dir, threshold):
    """Loads an image, performs recognition, saves annotated image and JSON."""
    print(f"Resim işleniyor: {image_path}")
    try:
        img_pil = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"HATA: Resim dosyası açılamadı: {image_path} - {e}")
        return

    start_time = time.time()
    annotated_img, results_list = recognize_faces_in_frame(img_pil, threshold)
    end_time = time.time()
    print(f"Resim işleme süresi: {end_time - start_time:.2f} saniye")


    output_image_filename = f"processed_{Path(image_path).name}"
    output_image_path = os.path.join(output_dir, output_image_filename)
    try:
        annotated_img.save(output_image_path)
        print(f"İşlenmiş resim kaydedildi: {output_image_path}")
    except Exception as e:
        print(f"HATA: İşlenmiş resim kaydedilemedi: {e}")


    json_data = {
        "source_file": str(Path(image_path).resolve()),
        "processing_timestamp": datetime.datetime.now().isoformat(),
        "threshold": threshold,
        "duration_seconds": round(end_time - start_time, 2),
        "recognitions": results_list
    }


    output_json_path = os.path.join(output_dir, "recognition_results.json")
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
        print(f"JSON sonuçları kaydedildi: {output_json_path}")
    except Exception as e:
        print(f"HATA: JSON sonuçları kaydedilemedi: {e}")


def process_video(video_path, output_dir, threshold):
    """Loads a video, performs recognition frame by frame, saves annotated video and JSON."""
    print(f"Video işleniyor: {video_path}")
    start_time_total = time.time()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"HATA: Video dosyası açılamadı: {video_path}")
        return


    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0: fps = 30.0 
    print(f"Video: {width}x{height} @ {fps:.2f} FPS, Kare Sayısı: {total_frames if total_frames > 0 else 'Bilinmiyor'}")


    output_video_filename = f"processed_{Path(video_path).stem}.mp4" 
    output_video_path = os.path.join(output_dir, output_video_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    try:
        out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if not out_writer.isOpened(): raise IOError("VideoWriter açılamadı.")
    except Exception as e:
        print(f"HATA: VideoWriter oluşturulamadı ({output_video_path}): {e}")
        cap.release()
        return


    frame_number = 0
    all_frame_results = []
    PROCESS_EVERY_N_FRAMES = 4
   

    while True:
        ret, frame_cv2_bgr = cap.read()
        if not ret:
            break 

        frame_number += 1
        print(f"\rİşlenen Kare: {frame_number}/{total_frames if total_frames > 0 else '?'}", end="")

        if frame_number % PROCESS_EVERY_N_FRAMES == 0:
            frame_pil_rgb = Image.fromarray(cv2.cvtColor(frame_cv2_bgr, cv2.COLOR_BGR2RGB))
            annotated_pil_frame, frame_recognition_list = recognize_faces_in_frame(frame_pil_rgb, threshold)

            if frame_recognition_list:
                 timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC) 
                 all_frame_results.append({
                     "frame_number": frame_number,
                     "timestamp_ms": round(timestamp_ms) if timestamp_ms else None,
                     "recognitions": frame_recognition_list
                 })


            output_cv2_frame = cv2.cvtColor(np.array(annotated_pil_frame), cv2.COLOR_RGB2BGR)
        else:
            output_cv2_frame = frame_cv2_bgr
          

        out_writer.write(output_cv2_frame)


    print("\nVideo işleme tamamlandı.")
    cap.release()
    out_writer.release()
    end_time_total = time.time()
    total_duration = end_time_total - start_time_total
    print(f"Toplam video işleme süresi: {total_duration:.2f} saniye")
    print(f"İşlenmiş video kaydedildi: {output_video_path}")



    json_data = {
        "source_file": str(Path(video_path).resolve()),
        "processing_timestamp": datetime.datetime.now().isoformat(),
        "threshold": threshold,
        "duration_seconds": round(total_duration, 2),
        "processed_frames": frame_number,
        "frames_with_recognitions": len(all_frame_results),
        "frames": all_frame_results 
    }

    output_json_path = os.path.join(output_dir, "recognition_results.json")
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
        print(f"JSON sonuçları kaydedildi: {output_json_path}")
    except Exception as e:
        print(f"HATA: JSON sonuçları kaydedilemedi: {e}")


def main():
    

    input_path = "project-02/gelen_fotograflar/Human/detected/face_f180_d0_20250513_104155_437083.png"
    threshold = 0.4

    if not os.path.exists(input_path):
        print(f"HATA: Giriş dosyası bulunamadı: {input_path}")
        return

    try:
        initialize_models()
        load_existing_database() 
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Başlatma hatası: {e}")
        print("Script sonlandırılıyor.")
        return 

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(RESULTS_BASE_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Çıktı dizini: {output_dir}")

   
    file_ext = Path(input_path).suffix.lower()
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

    if file_ext in image_extensions:
        process_image(input_path, output_dir, threshold)
    elif file_ext in video_extensions:
        process_video(input_path, output_dir, threshold)
    else:
        print(f"HATA: Desteklenmeyen dosya uzantısı: {file_ext}")
     
        try:
             if not os.listdir(output_dir): 
                  os.rmdir(output_dir)
        except OSError: pass


if __name__ == '__main__':
    main()