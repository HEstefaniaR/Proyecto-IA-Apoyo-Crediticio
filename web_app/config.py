import os
from dotenv import load_dotenv

load_dotenv()

# Botpress
BOTPRESS_TOKEN = os.getenv('BOTPRESS_TOKEN', '')
BOTPRESS_WORKSPACE_ID = os.getenv('BOTPRESS_WORKSPACE_ID', '')
BOTPRESS_BOT_ID = os.getenv('BOTPRESS_BOT_ID', '')
BOTPRESS_TABLE_NAME = os.getenv('BOTPRESS_TABLE_NAME')

# Flask
FLASK_ENV = os.getenv('FLASK_ENV', 'development')
SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'dev-key-insegura')
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 5001))

# Datos locales
DATA_DIR = os.getenv('DATA_DIR', './data')
CLIENTES_FILE = os.path.join(DATA_DIR, os.getenv('CLIENTES_FILE', 'clientes_data.json'))

# Seguridad CORS
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000').split(',')
CORS_ENABLED = os.getenv('CORS_ENABLED', 'true').lower() == 'true'

# Rutas modelos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BD_PATH = os.path.join(BASE_DIR, 'bd', 'bd_IA.xlsx')
MODELO1_PATH = os.path.join(BASE_DIR, 'modelos', 'modelo1_random_forest.pkl')
MODELO2_PATH = os.path.join(BASE_DIR, 'modelos', 'modelo2_XGBoost.pkl')

# Crear directorio de datos si no existe
os.makedirs(DATA_DIR, exist_ok=True)

# Verificar que archivos críticos existen
if not os.path.exists(BD_PATH):
    print(f"⚠️  ADVERTENCIA: No se encontró {BD_PATH}")
if not os.path.exists(MODELO1_PATH):
    print(f"⚠️  ADVERTENCIA: No se encontró {MODELO1_PATH}")
if not os.path.exists(MODELO2_PATH):
    print(f"⚠️  ADVERTENCIA: No se encontró {MODELO2_PATH}")