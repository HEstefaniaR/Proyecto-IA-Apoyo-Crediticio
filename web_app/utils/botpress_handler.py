import json
import os
import requests
from datetime import datetime
from config import (
    BOTPRESS_TOKEN, BOTPRESS_WORKSPACE_ID, BOTPRESS_BOT_ID,
    BOTPRESS_TABLE_NAME, CLIENTES_FILE
)

class BotpressHandler:
    def __init__(self):
        """Inicializa handler de Botpress"""
        self.token = BOTPRESS_TOKEN
        self.workspace_id = BOTPRESS_WORKSPACE_ID
        self.bot_id = BOTPRESS_BOT_ID
        self.table_name = BOTPRESS_TABLE_NAME
        self.data_file = CLIENTES_FILE
        
        self.base_url = "https://api.botpress.cloud/v1"
        # Dejamos las cabeceras base listas para reutilizar si es necesario
        self.headers = {
            'Authorization': f'Bearer {self.token}',
            'x-bot-id': self.bot_id,
            'Content-Type': 'application/json'
        }

    def descargar_tabla(self):
        """
        Descarga la tabla de Botpress usando el método POST oficial de búsqueda
        y la guarda localmente. Mantiene el estado 'cerrado' de clientes anteriores.
        
        Returns:
            list: Lista de clientes con datos fusionados
        """
        try:
            print(f"[Botpress] Descargando tabla ID: '{self.table_name}'...")
            
            # 1. Cambiado al endpoint de búsqueda exigido por Botpress para peticiones POST
            url = f"{self.base_url}/tables/{self.table_name}/rows/find"
            
            # 2. Cabeceras con x-bot-id tal como funcionó en el test exitoso
            headers = {
                'Authorization': f'Bearer {self.token}',
                'x-bot-id': self.bot_id,
                'Content-Type': 'application/json'
            }
            
            # 3. Payload estructurado para evitar el error 'InvalidPayload'
            payload = {
                'limit': 500,
                'offset': 0
            }
            
            # 4. Cambiado requests.get por requests.post
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code != 200:
                error_detail = response.text
                print(f"[ERROR] Botpress API respondió {response.status_code}")
                print(f"[ERROR] Detalles: {error_detail}")
                raise Exception(f"Botpress API error {response.status_code}: {error_detail}")
            
            data = response.json()
            nueva_data = data.get('rows', [])
            
            print(f"[Botpress] Se descargaron {len(nueva_data)} registros con éxito.")
            
            # Cargar datos anteriores (para mantener estado "cerrado")
            datos_anteriores = self._cargar_localmente()
            estado_cerrado = {
                str(c.get('cedula', '')).strip(): c.get('cerrado', False) 
                for c in datos_anteriores
            }
            
            # Fusionar: nueva data + estado anterior
            for cliente in nueva_data:
                cedula_str = str(cliente.get('cedula', '')).strip()
                
                # Mantener estado cerrado anterior
                if cedula_str in estado_cerrado:
                    cliente['cerrado'] = estado_cerrado[cedula_str]
                else:
                    cliente['cerrado'] = False
                
                # Agregar timestamp de sincronización
                cliente['ultima_sync'] = datetime.now().isoformat()
            
            # Guardar localmente
            self._guardar_localmente(nueva_data)
            print(f"[Botpress] ✓ Tabla guardada localmente en {self.data_file}")
            
            return nueva_data
            
        except requests.exceptions.Timeout:
            print(f"[ERROR] Timeout conectando con Botpress (30s)")
            return self._cargar_localmente()
        except requests.exceptions.ConnectionError as e:
            print(f"[ERROR] No se pudo conectar con Botpress: {e}")
            return self._cargar_localmente()
        except Exception as e:
            print(f"[ERROR] Al descargar tabla Botpress: {e}")
            return self._cargar_localmente()

    def _cargar_localmente(self):
        """Carga clientes del archivo JSON local"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"[WARN] Error cargando datos locales: {e}")
        return []

    def _guardar_localmente(self, datos):
        """Guarda clientes en archivo JSON local"""
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(datos, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[ERROR] Al guardar datos locales: {e}")
            raise

    def marcar_cerrado(self, cedula):
        """Marca un cliente como cerrado"""
        try:
            datos = self._cargar_localmente()
            cedula_str = str(cedula).strip()
            
            for cliente in datos:
                if str(cliente.get('cedula', '')).strip() == cedula_str:
                    cliente['cerrado'] = True
                    self._guardar_localmente(datos)
                    print(f"[Botpress] ✓ Cliente {cedula_str} marcado como cerrado")
                    return True
            
            print(f"[WARN] Cliente {cedula_str} no encontrado")
            return False
        except Exception as e:
            print(f"[ERROR] Al marcar cerrado: {e}")
            return False

    def marcar_abierto(self, cedula):
        """Reabre un cliente cerrado"""
        try:
            datos = self._cargar_localmente()
            cedula_str = str(cedula).strip()
            
            for cliente in datos:
                if str(cliente.get('cedula', '')).strip() == cedula_str:
                    cliente['cerrado'] = False
                    self._guardar_localmente(datos)
                    print(f"[Botpress] ✓ Cliente {cedula_str} reabierto")
                    return True
            
            print(f"[WARN] Cliente {cedula_str} no encontrado")
            return False
        except Exception as e:
            print(f"[ERROR] Al marcar abierto: {e}")
            return False

    def obtener_clientes(self, solo_activos=True):
        """Obtiene clientes del archivo local"""
        datos = self._cargar_localmente()
        if solo_activos:
            return [c for c in datos if not c.get('cerrado', False)]
        return datos