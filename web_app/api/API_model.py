from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import joblib
import os
import numpy as np
from datetime import datetime
import json

# Importar configuración y utilidades
from config import (
    FLASK_ENV, SECRET_KEY, HOST, PORT, 
    ALLOWED_ORIGINS, CORS_ENABLED,
    BD_PATH, MODELO1_PATH, MODELO2_PATH
)
from utils.botpress_handler import BotpressHandler

app = Flask(__name__, static_folder='../public', static_url_path='')

@app.route('/')
def serve_dashboard():
    # Esto busca el archivo dashboard.html dentro de la carpeta public y lo muestra
    return send_from_directory(app.static_folder, 'dashboard.html')

app.config['SECRET_KEY'] = SECRET_KEY

# CORS
if CORS_ENABLED:
    CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}})

# Inicializar Botpress
bp_handler = BotpressHandler()

# ===== CARGA DE MODELOS =====
try:
    m1 = joblib.load(MODELO1_PATH)
    m2_bundle = joblib.load(MODELO2_PATH)
    m2 = m2_bundle['mejor_modelo']
    oe_tipo = m2_bundle['le_tipo'] 
    oe_pag = m2_bundle['le_pag']
    le_target = m2_bundle.get('le_target')
    scaler_m2 = m2_bundle.get('scaler')
    print("✓ Modelos cargados correctamente")
except Exception as e:
    print(f"✗ Error cargando modelos: {e}")

# ===== FUNCIONES INTERNAS =====
# ===== FUNCIONES INTERNAS =====
def buscar_historial_cliente(cedula):
    try:
        df_bd = pd.read_excel(BD_PATH)
        df_bd.columns = df_bd.columns.str.strip().str.upper()

        target_col = 'ID_CLIENTE'
        if target_col not in df_bd.columns:
            return {'dias_desde_ultimo_credito': 9999, 'num_creditos_totales': 0, 'es_cliente_nuevo': True}

        # Normalizar: convertir todo a string entero sin decimales
        def normalizar(val):
            try:
                return str(int(float(str(val).strip())))
            except Exception:
                return str(val).strip().upper()

        df_bd[target_col] = df_bd[target_col].apply(normalizar)
        cedula_norm = normalizar(cedula)

        cliente = df_bd[df_bd[target_col] == cedula_norm]

        if cliente.empty:
            return {'dias_desde_ultimo_credito': 9999, 'num_creditos_totales': 0, 'es_cliente_nuevo': True}

        num_creditos = len(cliente)
        col_fecha = next((c for c in ['FECHA', 'FECHA_CREDITO', 'FECHA_DESEMBOLSO'] if c in df_bd.columns), None)

        if col_fecha:
            ultima_fecha = pd.to_datetime(cliente[col_fecha], errors='coerce').max()
            if pd.isna(ultima_fecha):
                dias_desde_ultimo = 9999
            else:
                dias_desde_ultimo = (datetime.now() - ultima_fecha).days
        else:
            dias_desde_ultimo = 9999

        return {
            'dias_desde_ultimo_credito': int(dias_desde_ultimo),
            'num_creditos_totales': int(num_creditos),
            'es_cliente_nuevo': False
        }

    except Exception as e:
        # Solo dejamos este print en caso de que ocurra un error real con el archivo
        print(f"[BD ERROR] Excepción en buscar_historial_cliente: {e}")
        return {'dias_desde_ultimo_credito': 9999, 'num_creditos_totales': 0, 'es_cliente_nuevo': True}



PAGADURIAS_PENSION = {'COLPENSIONES', 'ARP POSITIVA', 'FOPEP', 'FIDUPREVISORA', 'EJERCITO NACIONAL'}

def evaluar_reglas_negocio(edad, monto, plazo, tipo, pagaduria):
    resultados = {}
    
    razones_b = []
    if edad < 18 or edad > 84: 
        razones_b.append(f"Edad {edad} fuera de rango (18-84)")
    if tipo != 'REFINANCIACION' and plazo < 49: 
        razones_b.append(f"Plazo {plazo}m insuficiente para crédito nuevo")
    resultados['BUSINESS'] = {'ok': len(razones_b) == 0, 'razones': razones_b}

    razones_e = []
    if edad < 20 or edad > 81: 
        razones_e.append(f"Edad {edad} fuera de rango (20-81)")
    if monto < 1500000: 
        razones_e.append("Monto menor al mínimo $1.5M")
    resultados['EXCELCREDIT'] = {'ok': len(razones_e) == 0, 'razones': razones_e}

    razones_p = []
    pag_ok = any(p in pagaduria.upper() for p in PAGADURIAS_PENSION)
    if not pag_ok: 
        razones_p.append("Pagaduría no cubierta (requiere pensión pública)")
    if edad > 75: 
        razones_p.append("Edad supera máximo de 75 años")
    
    res_p = {'ok': len(razones_p) == 0, 'razones': razones_p}
    resultados['COPFINANCIAR'] = res_p
    resultados['PRONALCREDIT'] = res_p
    resultados['COOPIDESARROLLO'] = res_p
    
    return resultados

def predecir_oportunidad(datos_chatbot, historial):
    try:
        edad = float(datos_chatbot.get('edad', 0))
        monto = float(datos_chatbot.get('monto', 0))
        plazo = float(datos_chatbot.get('plazo', 0))
        tipo = str(datos_chatbot.get('tipo_credito', 'OTROS')).upper().strip()
        pag = str(datos_chatbot.get('pagaduria', 'OTROS')).upper().strip()

        analisis_reglas = evaluar_reglas_negocio(edad, monto, plazo, tipo, pag)
        
        def safe_encode(encoder, valor, col_name):
            try:
                temp_df = pd.DataFrame([valor], columns=[col_name])
                return encoder.transform(temp_df)[0][0]
            except:
                return 0.0

        tipo_encoded = safe_encode(oe_tipo, tipo, 'TIPO DE CREDITO')
        pag_encoded = safe_encode(oe_pag, pag, 'PAGADURIA')

        df2 = pd.DataFrame([[edad, monto, plazo, tipo_encoded, pag_encoded]], 
                           columns=['EDAD_AL_PRESTAMO', 'MONTO', 'PLAZO', 'TIPO DE CREDITO', 'PAGADURIA'])
        
        X_input = scaler_m2.transform(df2) if scaler_m2 else df2
        probs = m2.predict_proba(X_input)[0]
        clases = le_target.classes_

        ranking = []
        for i, coop in enumerate(clases):
            regla = analisis_reglas.get(coop, {'ok': True, 'razones': []})
            ranking.append({
                'cooperativa': str(coop),
                'prob_ml': round(float(probs[i]) * 100, 2),
                'elegible_reglas': regla['ok'],
                'razones_rechazo': regla['razones']
            })

        ranking = sorted(ranking, key=lambda x: x['prob_ml'], reverse=True)
        
        df1 = pd.DataFrame([[historial['dias_desde_ultimo_credito'], monto, plazo, historial['num_creditos_totales'], edad]],
                           columns=['dias_desde_ultimo_credito', 'MONTO', 'PLAZO', 'num_creditos_totales', 'EDAD_AL_PRESTAMO'])
        prob_aprob = m1.predict_proba(df1)[0][1]

        mejor_elegible = next((c for c in ranking if c['elegible_reglas']), None)
        if mejor_elegible is None and len(ranking) > 0:
            mejor_elegible = ranking[0]

        return {
            'prob_aprobacion': round(float(prob_aprob) * 100, 2),
            'ranking_cooperativas': ranking,
            'mejor_opcion_elegible': mejor_elegible,
            'historial_cliente': historial 
        }
    except Exception as e:
        print(f"Error en predecir_oportunidad: {e}")
        raise e

# ===== ENDPOINTS =====

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'environment': FLASK_ENV,
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/perfilar_cliente', methods=['POST'])
def perfilar_cliente():
    """
    🔥 ENDPOINT PRINCIPAL 🔥
    
    Descarga la tabla de Botpress, la procesa con los modelos
    y devuelve la lista de clientes con resultados.
    
    POST /api/perfilar_cliente
    """
    try:
        print("\n" + "="*70)
        print("INICIANDO PERFILACIÓN DE CLIENTES")
        print("="*70)
        
        # 1. Descargar tabla de Botpress
        clientes_raw = bp_handler.descargar_tabla()
        
        if not clientes_raw:
            return jsonify({'error': 'No hay clientes en la tabla'}), 400
        
        # 2. Procesar cada cliente
        resultados = []
        for i, cliente in enumerate(clientes_raw, 1):
            try:
                cedula = str(cliente.get('cedula', '')).strip()
                nombre = str(cliente.get('nombre', '')).strip()
                
                if not cedula:
                    print(f"  ⚠️  [{i}/{len(clientes_raw)}] Omitiendo: sin cédula")
                    continue
                
                # Buscar historial
                historial = buscar_historial_cliente(cedula)
                
                # Predecir
                resultado = predecir_oportunidad(cliente, historial)
                
                # Agregar a resultado final
                resultados.append({
                    'cedula': cedula,
                    'nombre': nombre,
                    'celular': cliente.get('celular', ''),
                    'edad': cliente.get('edad'),
                    'monto': cliente.get('monto'),
                    'plazo': cliente.get('plazo'),
                    'tipo_credito': cliente.get('tipo_credito'),
                    'pagaduria': cliente.get('pagaduria'),
                    'cerrado': cliente.get('cerrado', False),
                    'ultima_sync': cliente.get('ultima_sync'),
                    'prediccion': resultado
                })
                
                print(f"  ✓ [{i}/{len(clientes_raw)}] {nombre} ({cedula}) → {resultado['prob_aprobacion']}%")
                
            except Exception as e:
                print(f"  ✗ [{i}/{len(clientes_raw)}] Error procesando {cedula}: {e}")
        
        # Ordenar por viabilidad descendente
        resultados.sort(key=lambda x: x['prediccion']['prob_aprobacion'], reverse=True)
        
        print(f"\n✅ Perfilación completada: {len(resultados)} clientes procesados")
        print("="*70 + "\n")
        
        return jsonify({
            'status': 'ok',
            'total_clientes': len(resultados),
            'timestamp': datetime.now().isoformat(),
            'clientes': resultados
        }), 200
        
    except Exception as e:
        print(f"\n❌ ERROR en perfilar_cliente: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clientes', methods=['GET'])
def obtener_clientes():
    """
    GET /api/clientes?solo_activos=true
    
    Devuelve clientes (solo activos por defecto)
    """
    try:
        solo_activos = request.args.get('solo_activos', 'true').lower() == 'true'
        clientes = bp_handler.obtener_clientes(solo_activos=solo_activos)
        
        return jsonify({
            'status': 'ok',
            'total': len(clientes),
            'clientes': clientes
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cliente/<cedula>/cerrar', methods=['POST'])
def cerrar_cliente(cedula):
    try:
        data = request.get_json(silent=True) or {}
        motivo = data.get('motivo', 'perdido')  # 'perdido' o 'ganado'
        if bp_handler.marcar_cerrado(cedula):
            return jsonify({'status': 'ok', 'mensaje': f'Cliente {cedula} cerrado ({motivo})', 'motivo': motivo}), 200
        else:
            return jsonify({'error': 'Cliente no encontrado'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cliente/<cedula>/asesor', methods=['POST'])
def asignar_asesor(cedula):
    try:
        data = request.get_json(silent=True) or {}
        asesor = data.get('asesor', '')
        # Guardar en memoria (o puedes persistir en un archivo/BD)
        if not hasattr(app, 'asesores'):
            app.asesores = {}
        app.asesores[cedula] = asesor
        return jsonify({'status': 'ok', 'cedula': cedula, 'asesor': asesor}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/asesores', methods=['GET'])
def obtener_asesores():
    try:
        asesores = getattr(app, 'asesores', {})
        return jsonify({'status': 'ok', 'asesores': asesores}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cliente/<cedula>/abrir', methods=['POST'])
def abrir_cliente(cedula):
    """
    POST /api/cliente/{cedula}/abrir
    
    Reabre un cliente cerrado
    """
    try:
        if bp_handler.marcar_abierto(cedula):
            return jsonify({'status': 'ok', 'mensaje': f'Cliente {cedula} reabierto'}), 200
        else:
            return jsonify({'error': 'Cliente no encontrado'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(
        host=HOST,
        port=PORT,
        debug=(FLASK_ENV == 'development'),
        use_reloader=(FLASK_ENV == 'development')
    )