import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import HOST, PORT, FLASK_ENV
from api.API_model import app
from flask import send_from_directory

@app.route('/')
def index():
    return send_from_directory('public', 'dashboard.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('public', path)

if __name__ == '__main__':
    print(f"\n{'='*70}")
    print(f"🚀 Apoyo Crediticio - API")
    print(f"Environment: {FLASK_ENV}")
    print(f"Host: {HOST}:{PORT}")
    print(f"Dashboard: http://{HOST}:{PORT}")
    print(f"{'='*70}\n")
    
    app.run(host=HOST, port=PORT)