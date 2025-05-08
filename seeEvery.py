import gradio as gr
from ultralytics import YOLO
from deepface import DeepFace
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import json
import cv2
import time 
import shutil 
import pandas as pd 
import tempfile

# --- Ayarlar ---
K_SHOTS_ENROLLMENT_MANUAL = 3
MAX_IMAGES_PER_PERSON_FS = 150
EMBEDDING_MODEL_NAME = 'Facenet512' # Facenet512
YOLO_FACE_DETECTION_MODEL = 'project-02/yolov11m-face.pt' 
DEEPFACE_DETECTOR_BACKEND_FOR_CROPS = 'skip'
DEFAULT_DISTANCE_THRESHOLD = 0.50 
PROTOTYPES_FILE = 'face_prototypes_database_v4_facenet512.json' 
DEFAULT_DATASET_BASE_PATH = "project-02/gelen_fotograflar/Human" 

try:
    FONT_PATH = "arial.ttf" if os.name == 'nt' else "DejaVuSans.ttf"
    ImageFont.truetype(FONT_PATH, 15)
except IOError:
    FONT_PATH = None
    print(f"Uyarı: Belirtilen font ('{FONT_PATH if FONT_PATH else 'arial.ttf/DejaVuSans.ttf'}') bulunamadı. Varsayılan font kullanılacak.")


enrolled_prototypes = {} # {'isim': {'prototype_vector': np.array, 'num_samples': int, 'source': 'manual'/'filesystem'}}
yolo_model = None
deepface_ready = False


def initialize_models():
    global yolo_model, deepface_ready
    if yolo_model is None:
        try:
            yolo_model = YOLO(YOLO_FACE_DETECTION_MODEL)
            print(f"YOLO modeli '{YOLO_FACE_DETECTION_MODEL}' başarıyla yüklendi.")
        except Exception as e:
            print(f"HATA: YOLO modeli yüklenemedi: {e}")
    if not deepface_ready:
        try:
            dummy_img_for_warmup = np.zeros((64, 64, 3), dtype=np.uint8)
            DeepFace.represent(dummy_img_for_warmup, model_name=EMBEDDING_MODEL_NAME, enforce_detection=False, detector_backend=DEEPFACE_DETECTOR_BACKEND_FOR_CROPS)
            deepface_ready = True
            print(f"DeepFace embedding modeli '{EMBEDDING_MODEL_NAME}' kullanıma hazır.")
        except Exception as e:
            print(f"HATA: DeepFace modeli '{EMBEDDING_MODEL_NAME}' yüklenemedi/kullanılamadı: {e}")

def load_font(size=15):
    if FONT_PATH:
        try:
            return ImageFont.truetype(FONT_PATH, size)
        except IOError: pass
    try: return ImageFont.load_default(size=size)
    except TypeError: return ImageFont.load_default()

def save_prototypes_to_json():
    global enrolled_prototypes
    try:
        with open(PROTOTYPES_FILE, 'w', encoding='utf-8') as f:
            json_friendly_prototypes = {
                name: {
                    'prototype_vector': data['prototype_vector'].tolist(),
                    'num_samples': data['num_samples'],
                    'source': data.get('source', 'unknown')
                } for name, data in enrolled_prototypes.items()
            }
            json.dump(json_friendly_prototypes, f, indent=4, ensure_ascii=False)
        print(f"Prototip veritabanı '{PROTOTYPES_FILE}' dosyasına kaydedildi.")
        return f"{len(enrolled_prototypes)} kişi kayıtlı. Veritabanı güncellendi."
    except Exception as e:
        print(f"HATA: Prototip veritabanı kaydedilirken: {e}")
        return f"HATA: Veritabanı kaydedilemedi: {e}"

def load_prototypes_from_json():
    global enrolled_prototypes
    if os.path.exists(PROTOTYPES_FILE):
        try:
            with open(PROTOTYPES_FILE, 'r', encoding='utf-8') as f:
                json_friendly_prototypes = json.load(f)
                temp_prototypes = {}
                for name, data in json_friendly_prototypes.items():
                    temp_prototypes[name] = {
                        'prototype_vector': np.array(data['prototype_vector']),
                        'num_samples': data['num_samples'],
                        'source': data.get('source', 'manual')
                    }
                enrolled_prototypes = temp_prototypes 
            print(f"Prototip veritabanı '{PROTOTYPES_FILE}' dosyasından yüklendi ({len(enrolled_prototypes)} kişi).")
        except Exception as e:
            enrolled_prototypes = {} 
            print(f"HATA: Prototip veritabanı yüklenirken: {e}")
    else:
        enrolled_prototypes = {}
        print(f"'{PROTOTYPES_FILE}' bulunamadı. Kayıtlı kişi yok (JSON).")

def get_face_crop_from_image_pil(image_pil, yolo_detector_instance):
    if yolo_detector_instance is None: return None
    try:
        results = yolo_detector_instance(image_pil, verbose=False, conf=0.3)
    except Exception as e:
        print(f"YOLO ile yüz tespiti sırasında hata: {e}")
        return None
        
    best_box = None; max_area = 0
    for result in results:
        for box_data in result.boxes:
            if box_data.conf[0] > 0.3:
                x1, y1, x2, y2 = map(int, box_data.xyxy[0].tolist())
                area = (x2 - x1) * (y2 - y1)
                if area > max_area: max_area = area; best_box = (x1, y1, x2, y2)
    if best_box:
        img_width, img_height = image_pil.size
        x1_c,y1_c,x2_c,y2_c = max(0,best_box[0]),max(0,best_box[1]),min(img_width,best_box[2]),min(img_height,best_box[3])
        if x1_c < x2_c and y1_c < y2_c: return image_pil.crop((x1_c, y1_c, x2_c, y2_c))
    return None

def get_embedding_from_face_pil(face_crop_pil):
    global deepface_ready
    if not deepface_ready or face_crop_pil is None or face_crop_pil.width < 20 or face_crop_pil.height < 20: return None
    try:
        face_crop_numpy_rgb = np.array(face_crop_pil.convert('RGB'))
        face_crop_numpy_bgr = cv2.cvtColor(face_crop_numpy_rgb, cv2.COLOR_RGB2BGR)
        embedding_objs = DeepFace.represent(
            img_path=face_crop_numpy_bgr, model_name=EMBEDDING_MODEL_NAME,
            enforce_detection=False, detector_backend=DEEPFACE_DETECTOR_BACKEND_FOR_CROPS
        )
        if embedding_objs and len(embedding_objs) > 0: return np.array(embedding_objs[0]["embedding"])
        return None
    except Exception as e:
        if "Face could not be detected" not in str(e) and "Singleton array array" not in str(e) and "expected BGR image" not in str(e).lower():
             print(f"Embedding çıkarılırken hata: {e} (Yüz boyutu: {face_crop_pil.size})")
        return None

def calculate_cosine_distance_numpy(vec1, vec2):
    if vec1 is None or vec2 is None: return float('inf')
    norm_vec1 = np.linalg.norm(vec1); norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0: return float('inf')
    return 1 - (np.dot(vec1, vec2) / (norm_vec1 * norm_vec2))

def scan_filesystem_and_enroll_interface(dataset_base_path_str, progress=gr.Progress(track_tqdm=True)):
    global enrolled_prototypes, yolo_model
    if not dataset_base_path_str or not os.path.isdir(dataset_base_path_str):
        return "Hata: Geçersiz klasör yolu.", pd.DataFrame()
    if yolo_model is None or not deepface_ready:
        return "Hata: Ana modeller (YOLO/DeepFace) yüklenemedi.", pd.DataFrame()

    progress(0, desc="Klasörler taranıyor...")
    person_folders = [d for d in os.listdir(dataset_base_path_str) if os.path.isdir(os.path.join(dataset_base_path_str, d))]
    total_folders = len(person_folders)
    fs_prototypes_found = {}
    processed_person_count = 0

    for i, person_name in enumerate(person_folders):
        progress((i + 1) / total_folders, desc=f"'{person_name}' işleniyor...")
        person_dir = os.path.join(dataset_base_path_str, person_name)
        embeddings_list = []
        images_processed_count = 0
        image_files_in_folder = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for image_file in sorted(image_files_in_folder):
            if images_processed_count >= MAX_IMAGES_PER_PERSON_FS: break
            image_path = os.path.join(person_dir, image_file)
            try:
                img_pil = Image.open(image_path).convert("RGB")
                face_crop_pil = get_face_crop_from_image_pil(img_pil, yolo_model)
                if face_crop_pil:
                    embedding = get_embedding_from_face_pil(face_crop_pil)
                    if embedding is not None:
                        embeddings_list.append(embedding)
                        images_processed_count += 1
            except Exception as e:
                print(f"    Hata: '{image_file}' işlenirken ({person_name}): {e}")
        
        if embeddings_list:
            prototype = np.mean(embeddings_list, axis=0)
            fs_prototypes_found[person_name] = {
                'prototype_vector': prototype,
                'num_samples': len(embeddings_list),
                'source': 'filesystem'
            }
            processed_person_count += 1
    
    for name, data in fs_prototypes_found.items():
        enrolled_prototypes[name] = data # Eski FS kayıtlarını veya manuel olanları günceller/ekler
    
    save_prototypes_to_json()
    status_message = f"{processed_person_count} kişi dosya sisteminden işlendi/güncellendi. Toplam {len(enrolled_prototypes)} kayıtlı kişi."
    print(status_message)
    return status_message, get_enrolled_people_dataframe() # Güncellenmiş DataFrame'i döndür

# --- Manuel Kayıt ---
def enroll_person_manual_interface(person_name_manual, img1_pil, img2_pil, img3_pil):
    global enrolled_prototypes
    if not deepface_ready: return "Hata: DeepFace modeli hazır değil.", get_enrolled_people_dataframe()
    if not person_name_manual.strip(): return "Hata: Kişi adı boş olamaz.", get_enrolled_people_dataframe()

    images_pil = [img for img in [img1_pil, img2_pil, img3_pil] if img is not None]
    if len(images_pil) != K_SHOTS_ENROLLMENT_MANUAL:
        return f"Hata: Lütfen tam olarak {K_SHOTS_ENROLLMENT_MANUAL} adet yüz fotoğrafı yükleyin.", get_enrolled_people_dataframe()

    embeddings_list = []
    for i, img_pil in enumerate(images_pil):
        embedding = get_embedding_from_face_pil(img_pil)
        if embedding is not None: embeddings_list.append(embedding)
        else: return f"Hata: {i+1}. fotoğraftan yüz özelliği çıkarılamadı.", get_enrolled_people_dataframe()

    if len(embeddings_list) == K_SHOTS_ENROLLMENT_MANUAL:
        prototype = np.mean(embeddings_list, axis=0)
        enrolled_prototypes[person_name_manual] = {
            'prototype_vector': prototype,
            'num_samples': len(embeddings_list),
            'source': 'manual'
        }
        save_prototypes_to_json()
        return f"'{person_name_manual}' manuel olarak kaydedildi/güncellendi.", get_enrolled_people_dataframe()
    else: return "Hata: Yeterli geçerli yüz özelliği çıkarılamadı.", get_enrolled_people_dataframe()

# --- Kayıtlı Kişi Yönetimi ---
def get_enrolled_people_dataframe():
    global enrolled_prototypes
    if not enrolled_prototypes:
        return pd.DataFrame(columns=["İsim", "Örnek Sayısı", "Kaynak"])
    data = []
    for name, info in enrolled_prototypes.items():
        data.append([name, info['num_samples'], info.get('source', 'Bilinmiyor')])
    return pd.DataFrame(data, columns=["İsim", "Örnek Sayısı", "Kaynak"])

def get_enrolled_people_names():
    return list(enrolled_prototypes.keys())

def delete_person_interface(person_name_to_delete):
    global enrolled_prototypes
    if not person_name_to_delete:
        return "Silinecek kişi seçilmedi.", get_enrolled_people_dataframe(), gr.Dropdown(choices=get_enrolled_people_names())
    if person_name_to_delete in enrolled_prototypes:
        del enrolled_prototypes[person_name_to_delete]
        save_prototypes_to_json()
        msg = f"'{person_name_to_delete}' adlı kişi silindi."
        print(msg)
        return msg, get_enrolled_people_dataframe(), gr.Dropdown(choices=get_enrolled_people_names(), label="Silinecek Kişiyi Seçin")
    else:
        return f"Hata: '{person_name_to_delete}' adlı kişi bulunamadı.", get_enrolled_people_dataframe(), gr.Dropdown(choices=get_enrolled_people_names())


def process_frame_and_get_annotations(frame_pil, current_distance_threshold):
    global enrolled_prototypes, yolo_model
    if yolo_model is None or not deepface_ready: return frame_pil, "Hata: Modeller yüklenemedi.", []
    if not enrolled_prototypes: return frame_pil, "Kayıtlı kişi yok.", []

    try:
        results = yolo_model.predict(frame_pil, verbose=False, conf=0.4)
    except Exception as e:
        print(f"YOLO ile yüz tespiti sırasında hata (process_frame): {e}")
        return frame_pil, f"YOLO hatası: {e}", []

    annotated_frame_pil = frame_pil.copy()
    draw = ImageDraw.Draw(annotated_frame_pil)
    font_size = max(12, int(frame_pil.height / 45))
    font = load_font(size=font_size)
    raw_annotations = []; recognition_info_list = []

    for result_set in results:
        for box_data in result_set.boxes:
            x1,y1,x2,y2 = map(int, box_data.xyxy[0].tolist())
            face_crop_pil = frame_pil.crop((x1,y1,x2,y2))
            if face_crop_pil.width<20 or face_crop_pil.height<20: continue
            query_embedding = get_embedding_from_face_pil(face_crop_pil)
            identity="Bilinmeyen"; min_distance=float('inf'); color="red"; best_match_name_for_unknown="Yok"

            if query_embedding is not None:
                for name, data in enrolled_prototypes.items():
                    distance = calculate_cosine_distance_numpy(query_embedding, data['prototype_vector'])
                    if distance < min_distance: min_distance=distance; best_match_name_for_unknown=name
                if min_distance < current_distance_threshold:
                    identity=best_match_name_for_unknown; color="lime"
                    recognition_info_list.append(f"Tanınan: {identity} (Mesafe: {min_distance:.3f})")
                else:
                    recognition_info_list.append(f"Bilinmeyen (En yakın: {best_match_name_for_unknown}, Mesafe: {min_distance:.3f})")
            else:
                recognition_info_list.append(f"Yüzden özellik çıkarılamadı ({x1},{y1}).")
                min_distance=float('inf')
            
            label_text = f"{identity} ({min_distance:.2f})" if query_embedding is not None else "Özellik Yok"
            raw_annotations.append(((x1,y1,x2,y2), label_text, color))
            draw.rectangle([(x1,y1),(x2,y2)], outline=color, width=max(1,int(frame_pil.height/250)))
            try: text_bbox=draw.textbbox((x1,y1-font_size-2),label_text,font=font); text_w,text_h=text_bbox[2]-text_bbox[0],text_bbox[3]-text_bbox[1]
            except AttributeError: text_w,text_h=draw.textlength(label_text,font=font) if font else (70,10), font_size if font else 10
            bg_y1 = y1-text_h-4 if (y1-text_h-4)>0 else y1+2
            draw.rectangle([(x1,bg_y1),(x1+text_w+4,bg_y1+text_h+2)], fill=color)
            draw.text((x1+2,bg_y1),label_text, fill="black", font=font)
            
    summary_text = "\n".join(recognition_info_list) if recognition_info_list else "Anlamlı yüz bulunamadı/işlenemedi."
    return annotated_frame_pil, summary_text, raw_annotations

# --- Resimden Tanıma ---
def recognize_faces_image_interface(input_image_pil, current_distance_threshold):
    if input_image_pil is None: return None, "Hata: Giriş resmi alınamadı."
    annotated_pil_frame, summary_text, _ = process_frame_and_get_annotations(input_image_pil, float(current_distance_threshold))
    return annotated_pil_frame, summary_text

def recognize_faces_video_interface(video_path_input, current_distance_threshold, progress=gr.Progress(track_tqdm=True)):
    if video_path_input is None: return None, "Hata: Video dosyası yüklenmedi."
    video_path = video_path_input # Gradio geçici dosya yolu verir

    if not os.path.exists(video_path):
        return None, f"Hata: Video dosyası bulunamadı: {video_path}"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, f"Hata: Video dosyası açılamadı: {video_path}"

    fps = cap.get(cv2.CAP_PROP_FPS)
    # FPS 0 ise veya çok yüksekse makul bir değere ayarla (örn: 30)
    if fps == 0 or fps > 120:  # Bazı hatalı videolarda fps 0 veya çok yüksek olabilir
        print(f"Uyarı: Videonun FPS değeri ({fps}) geçersiz görünüyor. Varsayılan olarak 30 FPS kullanılacak.")
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Eğer width veya height 0 ise, ilk kareden okumayı dene
    if width == 0 or height == 0:
        ret_tmp, frame_tmp = cap.read()
        if ret_tmp:
            height, width, _ = frame_tmp.shape
            print(f"Uyarı: Video başlık bilgisi eksik, ilk kareden boyutlar alındı: {width}x{height}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Video okuyucuyu başa sar
        else:
            cap.release()
            return None, "Hata: Video boyutları okunamadı."


    # Python'un tempfile modülü ile güvenli bir geçici dosya oluştur
    # delete=False önemlidir, böylece Gradio dosyayı okuyana kadar silinmez.
    # Gradio dosyayı kendi önbelleğine aldıktan sonra bu geçici dosya önemsizleşir.
    # Gradio genellikle kendi geçici dosyalarını kendi yönetir.
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
        output_video_temp_path = tmpfile.name
    
    print(f"Geçici çıktı video dosyası: {output_video_temp_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # veya 'avc1' H.264 için
    out_writer = cv2.VideoWriter(output_video_temp_path, fourcc, fps, (width,height))
    
    if not out_writer.isOpened():
        cap.release()
        print(f"HATA: VideoWriter oluşturulamadı: {output_video_temp_path}")
        # Eğer dosya sisteminde yazma izni yoksa veya codec desteklenmiyorsa bu hata olabilir.
        # Farklı bir fourcc deneyebilirsiniz: fourcc = cv2.VideoWriter_fourcc(*'XVID') (daha yaygın)
        # Veya fourcc = 0 (sistem varsayılanını kullanır, platforma bağlı)
        # fourcc = cv2.VideoWriter_fourcc('M','J','P','G') # MJPEG, daha büyük dosyalar ama daha uyumlu
        return None, f"Hata: Çıktı videosu ({output_video_temp_path}) için VideoWriter oluşturulamadı. Codec veya yazma izni sorunu olabilir."


    frame_counter = 0
    last_known_raw_annotations = [] 
    all_summaries = []

    # total_frames 0 veya hatalıysa, progress.tqdm düzgün çalışmayabilir.
    # Bu durumda sadece desc gösterir.
    iterable_frames = range(total_frames) if total_frames > 0 else iter(int, 1) # Sonsuz döngü değil, cap.read() ile kontrol

    for i in progress.tqdm(iterable_frames, desc="Video işleniyor...", total=total_frames if total_frames > 0 else None):
        ret, frame_cv2_bgr = cap.read()
        if not ret: 
            if total_frames == 0 : break # total_frames bilinmiyorsa ve okuma bittiyse çık
            # Eğer total_frames biliniyorsa ve daha erken bittiyse, tqdm'in sonuna kadar gitmesine gerek yok
            print(f"Uyarı: Video beklenenden erken bitti. Okunan kare: {frame_counter}, beklenen: {total_frames}")
            break

        frame_counter += 1
        current_frame_pil_rgb = Image.fromarray(cv2.cvtColor(frame_cv2_bgr, cv2.COLOR_BGR2RGB))
        output_pil_frame_to_write = current_frame_pil_rgb.copy()

        if frame_counter % 4 == 0:
            annotated_pil_frame, summary, raw_annotations = process_frame_and_get_annotations(current_frame_pil_rgb, float(current_distance_threshold))
            if raw_annotations: last_known_raw_annotations = raw_annotations
            if summary and "Hata:" not in summary and "bulunmamaktadır" not in summary and "tespit edilemedi" not in summary:
                 all_summaries.append(f"Kare {frame_counter}: {summary}")
            output_pil_frame_to_write = annotated_pil_frame 
        else:
            if last_known_raw_annotations:
                draw = ImageDraw.Draw(output_pil_frame_to_write)
                font_size = max(12,int(height/45)); font = load_font(size=font_size)
                for (x1_ann,y1_ann,x2_ann,y2_ann),label_text,color in last_known_raw_annotations: # Değişken adları çakışmasın
                    draw.rectangle([(x1_ann,y1_ann),(x2_ann,y2_ann)], outline=color, width=max(1,int(height/250)))
                    try: text_bbox=draw.textbbox((x1_ann,y1_ann-font_size-2),label_text,font=font); text_w,text_h=text_bbox[2]-text_bbox[0],text_bbox[3]-text_bbox[1]
                    except AttributeError: text_w,text_h=draw.textlength(label_text,font=font) if font else (70,10), font_size if font else 10
                    bg_y1_ann = y1_ann-text_h-4 if (y1_ann-text_h-4)>0 else y1_ann+2 # Değişken adları
                    draw.rectangle([(x1_ann,bg_y1_ann),(x1_ann+text_w+4,bg_y1_ann+text_h+2)],fill=color)
                    draw.text((x1_ann+2,bg_y1_ann),label_text,fill="black",font=font)
        
        out_writer.write(cv2.cvtColor(np.array(output_pil_frame_to_write), cv2.COLOR_RGB2BGR))
    
    cap.release()
    out_writer.release()
    
    final_summary_text = "\n---\n".join(all_summaries) if all_summaries else "Videoda önemli bir tanıma yapılmadı veya tüm kareler atlandı/işlenemedi."
    if frame_counter == 0 and total_frames > 0 : # Hiç kare okunamadıysa
        print(f"HATA: Videodan hiç kare okunamadı: {video_path}")
        return None, "Hata: Video dosyası okunamadı veya boş."

    print(f"Video işleme tamamlandı. Çıktı: {output_video_temp_path}")
    return output_video_temp_path, final_summary_text

# --- Uygulama Başlangıcı ---
initialize_models()
load_prototypes_from_json() # JSON'dan yükle, sonra FS taraması bunu güncelleyebilir/üzerine yazabilir

# --- Gradio Arayüzü (Değişiklik Yok) ---
# ... (Gradio arayüz tanımı aynı kalacak) ...
# with gr.Blocks(theme=gr.themes.Monochrome(), title="Yüz Tanıma v4") as demo:
# ... (Tüm Gradio sekme ve bileşen tanımları öncekiyle aynı) ...

# Sadece tam olması için Gradio arayüzünü tekrar ekleyelim:
with gr.Blocks(theme=gr.themes.Monochrome(), title="Yüz Tanıma v4") as demo:
    gr.Markdown("# 🧔🏽 Yüz Tanıma Sistemi v4 (Gelişmiş Yönetim ve Ayarlar)")
    
    with gr.Tab("️🗃️ Veritabanı Yönetimi"):
        gr.Markdown("Kişi fotoğraflarının bulunduğu ana klasörü tarayarak veya manuel olarak yüz prototiplerini yönetin.")
        with gr.Row():
            with gr.Column(scale=2):
                db_path_input_manage = gr.Textbox(label="Veritabanı Ana Klasör Yolu (Dosya Sistemi Taraması İçin)", value=DEFAULT_DATASET_BASE_PATH)
                scan_db_button_manage = gr.Button("Dosya Sistemini Tara ve Prototipleri Güncelle/Ekle")
                db_status_output_manage = gr.Textbox(label="Tarama Durumu", interactive=False)
            with gr.Column(scale=3):
                gr.Markdown("#### Kayıtlı Kişiler")
                enrolled_list_df_manage = gr.DataFrame(value=get_enrolled_people_dataframe, label="Kayıtlı Kişiler Listesi", interactive=False)
                with gr.Row():
                    delete_person_dropdown_manage = gr.Dropdown(choices=get_enrolled_people_names(), label="Silinecek Kişiyi Seçin", interactive=True)
                    delete_person_button_manage = gr.Button("Seçili Kişiyi Sil")
                delete_status_output_manage = gr.Textbox(label="Silme Durumu", interactive=False)

        scan_db_button_manage.click(
            scan_filesystem_and_enroll_interface,
            inputs=[db_path_input_manage],
            outputs=[db_status_output_manage, enrolled_list_df_manage]
        ).then(lambda: gr.Dropdown(choices=get_enrolled_people_names()), outputs=delete_person_dropdown_manage)

        delete_person_button_manage.click(
            delete_person_interface,
            inputs=[delete_person_dropdown_manage],
            outputs=[delete_status_output_manage, enrolled_list_df_manage, delete_person_dropdown_manage]
        )

    with gr.Tab("🆕 Manuel Kişi Kaydet"):
        gr.Markdown(f"Tanınacak her kişi için **{K_SHOTS_ENROLLMENT_MANUAL} adet net yüz fotoğrafı** (kırpılmış) yükleyin.")
        person_name_manual_input_enroll = gr.Textbox(label="Kişinin Adı Soyadı")
        with gr.Row():
            manual_img_inputs_enroll = [gr.Image(type="pil", label=f"{i+1}. Yüz Fotoğrafı", sources=["upload"]) for i in range(K_SHOTS_ENROLLMENT_MANUAL)]
        manual_enroll_button_enroll = gr.Button("Bu Kişiyi Manuel Kaydet")
        manual_enroll_status_output_enroll = gr.Textbox(label="Manuel Kayıt Durumu", interactive=False)
        
        manual_enroll_button_enroll.click(
            enroll_person_manual_interface,
            inputs=[person_name_manual_input_enroll] + manual_img_inputs_enroll,
            outputs=[manual_enroll_status_output_enroll, enrolled_list_df_manage]
        ).then(lambda: gr.Dropdown(choices=get_enrolled_people_names()), outputs=delete_person_dropdown_manage)

    with gr.Tab("👁️‍🗨️ Yüz Tanıma (Resim/Video)"):
        gr.Markdown("Tanıma hassasiyetini ayarlayabilir, resim, webcam veya video ile yüz tanıma yapabilirsiniz.")
        distance_threshold_slider_rec = gr.Slider(minimum=0.1, maximum=1.0, value=DEFAULT_DISTANCE_THRESHOLD, step=0.01, label="Mesafe Eşiği (Düşük = Daha Katı Eşleşme)")
        
        with gr.Accordion("📷 Resimden / Webcam'den Tanıma", open=True):
            img_rec_input_rec = gr.Image(type="pil", label="Fotoğraf Yükle / Webcam Başlat", sources=["upload", "webcam", "clipboard"])
            img_rec_button_rec = gr.Button("Resimdeki/Webcam Anlık Görüntüsündeki Yüzleri Tanı") 
            with gr.Row():
                img_rec_output_img_rec = gr.Image(type="pil", label="Tanıma Sonucu (Resim)")
                img_rec_output_summary_rec = gr.Textbox(label="Tanıma Özeti (Resim)", interactive=False, lines=5)
            img_rec_button_rec.click(
                 recognize_faces_image_interface,
                 inputs=[img_rec_input_rec, distance_threshold_slider_rec],
                 outputs=[img_rec_output_img_rec, img_rec_output_summary_rec]
            )

        with gr.Accordion("📹 Videodan Tanıma", open=False):
            video_rec_input_vid = gr.Video(label="Video Dosyası Yükle")
            video_rec_button_vid = gr.Button("Videodaki Yüzleri Tanı")
            with gr.Row():
                video_rec_output_video_vid = gr.Video(label="Tanıma Sonucu (Video)")
                video_rec_output_summary_vid = gr.Textbox(label="Tanıma Özeti (Video)", interactive=False, lines=10)
            video_rec_button_vid.click(
                recognize_faces_video_interface,
                inputs=[video_rec_input_vid, distance_threshold_slider_rec],
                outputs=[video_rec_output_video_vid, video_rec_output_summary_vid]
            )
            
    gr.Markdown("---")
    gr.Markdown(f"Geliştirici: Google Gemini & AI Kullanıcısı | Modeller: {YOLO_FACE_DETECTION_MODEL} + DeepFace ({EMBEDDING_MODEL_NAME})")

if __name__ == '__main__':
    if not os.path.exists(DEFAULT_DATASET_BASE_PATH):
        try:
            os.makedirs(DEFAULT_DATASET_BASE_PATH)
            print(f"Örnek veritabanı klasörü '{DEFAULT_DATASET_BASE_PATH}' oluşturuldu.")
        except OSError as e: print(f"'{DEFAULT_DATASET_BASE_PATH}' klasörü oluşturulamadı: {e}")
            
    demo.launch(debug=False)
