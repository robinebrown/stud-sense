from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).parent.resolve()

# -------------------- Directory Paths --------------------
# Source meshes and assets
OBJS_DIR = BASE_DIR / 'objs'

# Synthetic dataset
DATASET_DIR = BASE_DIR / 'dataset'
IMAGES_DIR = DATASET_DIR / 'images'
LABELS_DIR = DATASET_DIR / 'labels'
TRAIN_IMAGES_DIR = IMAGES_DIR / 'train'
VAL_IMAGES_DIR = IMAGES_DIR / 'val'
TEST_IMAGES_DIR = IMAGES_DIR / 'test'
TRAIN_LABELS_DIR = LABELS_DIR / 'train'
VAL_LABELS_DIR = LABELS_DIR / 'val'
TEST_LABELS_DIR = LABELS_DIR / 'test'

# Embeddings
EMBEDDINGS_DIR = BASE_DIR / 'embeddings'
CANONICAL_IMAGES_DIR = EMBEDDINGS_DIR / 'canonical'
EMBEDDER_SCRIPTED = EMBEDDINGS_DIR / 'embedder_scripted.pt'
CANONICAL_FEATS = EMBEDDINGS_DIR / 'canonical_feats.npy'
CANONICAL_LABELS = EMBEDDINGS_DIR / 'canonical_labels.pkl'
FAISS_INDEX = EMBEDDINGS_DIR / 'faiss.index'

# YOLO model training & artifacts
YOLO_CONFIG = BASE_DIR / 'lego.yaml'
YOLO_BASE_MODEL = 'yolo11n.pt'
YOLO_RUNS = BASE_DIR / 'runs' / 'detect'
YOLO_TRAIN_RUN = YOLO_RUNS / 'train'
YOLO_WEIGHTS = YOLO_TRAIN_RUN / 'weights'
YOLO_TFLITE = YOLO_WEIGHTS / 'best_saved_model' / 'best_float32.tflite'

# -------------------- Hyperparameters --------------------
# YOLO detector
YOLO_EPOCHS = 50
YOLO_BATCH_SIZE = 16
YOLO_LR = 1e-3
YOLO_DEVICE = 'cuda'  # or 'cpu'

# Embedding extractor
EMBED_BATCH_SIZE = 32

# Logging
LOG_LEVEL = 'INFO'

# Synthetic renderer
# If you want to override which parts to generate, set PART_IDS below
# PART_IDS = ['3001', '3002', ...]
PART_IDS = "3001","3003","3020","3022","3040","54200","3665","3062","4740","15254",
"4865","32000","2780","3626","2555","6238","51239","88293","2817","30414"
