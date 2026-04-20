from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import numpy as np
from datetime import datetime
import json  # Importamos el módulo json estándar

app = Flask(__name__)

# ===== CONFIGURACION DE RUTAS =====
BASE_DIR = '/Users/estefania/Documents/proyecto IA Apoyo Crediticio'
BD_PATH = os.path.join(BASE_DIR, 'data/bd_IA.xlsx')
MODELO1_PATH = os.path.join(BASE_DIR, 'modelos/modelo1_random_forest.pkl')
MODELO2_PATH = os.path.join(BASE_DIR, 'modelos/modelo2_Gradient_Boosting.pkl')

# ===== CARGA DE MODELOS Y COMPONENTES =====
try:
    m1 = joblib.load(MODELO1_PATH)
    m2_bundle = joblib.load(MODELO2_PATH)
    m2 = m2_bundle['mejor_modelo']
    oe_tipo = m2_bundle['le_tipo'] 
    oe_pag = m2_bundle['le_pag']
    le_target = m2_bundle.get('le_target')
    scaler_m2 = m2_bundle.get('scaler')
    print("✓ Modelos y componentes cargados correctamente")
except Exception as e:
    print(f"✗ Error cargando componentes: {e}")

# ===== FUNCIONES INTERNAS =====

def buscar_historial_cliente(cedula):
    try:
        df_bd = pd.read_excel(BD_PATH)
        df_bd.columns = df_bd.columns.str.strip().str.upper()
        cliente = df_bd[df_bd['ID_CLIENTE'].astype(str) == str(cedula)]
        
        if cliente.empty:
            return {'dias_desde_ultimo_credito': 9999, 'num_creditos_totales': 0, 'es_cliente_nuevo': True}
        
        num_creditos = len(cliente)
        col_fecha = 'FECHA' if 'FECHA' in df_bd.columns else 'FECHA_CREDITO'
        if col_fecha in cliente.columns:
            ultima_fecha = pd.to_datetime(cliente[col_fecha]).max()
            dias_desde_ultimo = (datetime.now() - ultima_fecha).days
        else:
            dias_desde_ultimo = 9999
        return {'dias_desde_ultimo_credito': dias_desde_ultimo, 'num_creditos_totales': num_creditos, 'es_cliente_nuevo': False}
    except Exception as e:
        return {'dias_desde_ultimo_credito': 9999, 'num_creditos_totales': 0, 'es_cliente_nuevo': True}

def predecir_oportunidad(datos_chatbot, historial):
    # --- MODELO 1 ---
    df1 = pd.DataFrame([{
        'dias_desde_ultimo_credito': historial['dias_desde_ultimo_credito'],
        'MONTO': float(datos_chatbot['monto']),
        'PLAZO': float(datos_chatbot['plazo']),
        'num_creditos_totales': historial['num_creditos_totales'],
        'EDAD_AL_PRESTAMO': float(datos_chatbot.get('edad', 0))
    }])
    df1 = df1[['dias_desde_ultimo_credito', 'MONTO', 'PLAZO', 'num_creditos_totales', 'EDAD_AL_PRESTAMO']]
    prob_aprobacion = m1.predict_proba(df1)[0][1]

    # --- MODELO 2 ---
    def safe_encode(encoder, valor):
        try:
            # OrdinalEncoder espera un array 2D
            return encoder.transform([[str(valor).upper().strip()]])[0][0]
        except:
            return 0.0 

    df2 = pd.DataFrame([{
        'EDAD_AL_PRESTAMO': float(datos_chatbot.get('edad', 0)),
        'MONTO': float(datos_chatbot['monto']),
        'PLAZO': float(datos_chatbot['plazo']),
        'TIPO DE CREDITO': safe_encode(oe_tipo, datos_chatbot.get('tipo_credito', 'OTROS')),
        'PAGADURIA': safe_encode(oe_pag, datos_chatbot.get('pagaduria', 'OTROS'))
    }])
    df2 = df2[['EDAD_AL_PRESTAMO', 'MONTO', 'PLAZO', 'TIPO DE CREDITO', 'PAGADURIA']]
    
    if scaler_m2:
        df2_final = scaler_m2.transform(df2)
    else:
        df2_final = df2

    pred_idx = m2.predict(df2_final)[0]
    
    # Traducir índice a Cooperativa
    try:
        coop_nombre = le_target.inverse_transform([pred_idx])[0]
    except:
        coop_nombre = f"ID: {pred_idx}"
    
    try:
        prob_m2 = np.max(m2.predict_proba(df2_final)[0])
    except:
        prob_m2 = 0.5

    score_final = (prob_aprobacion * 0.7 + prob_m2 * 0.3) * 100

    return {
        'score_prioridad': round(score_final, 2),
        'prob_aprobacion': round(prob_aprobacion * 100, 2),
        'cooperativa': coop_nombre,
        'confianza_coop': round(prob_m2 * 100, 2)
    }

# ===== ENDPOINTS =====

@app.route('/api/evaluar_desde_archivo', methods=['POST'])
def evaluar_desde_archivo():
    try:
        req = request.json
        rel_path = req.get('archivo_path').replace('./', '').replace('../', '')
        path_archivo = os.path.join(BASE_DIR, rel_path)
        
        with open(path_archivo, 'r') as f:
            lineas = f.readlines()
            # CORRECCIÓN AQUÍ: usamos el módulo json estándar
            datos = json.loads(lineas[-1]) 
            
        historial = buscar_historial_cliente(datos['cedula'])
        res = predecir_oportunidad(datos, historial)
        
        # === IMPRESIÓN EN TERMINAL ===
        print("\n" + "═"*60)
        print("              RESUMEN DE EVALUACIÓN DE PROSPECTO")
        print("═"*60)
        print(f"PROSPECTO:      {datos.get('nombre', 'N/A').upper()}")
        print(f"CÉDULA:         {datos.get('cedula', 'N/A')}")
        print(f"EDAD:           {datos.get('edad', 'N/A')} años")
        print(f"PAGADURÍA:      {datos.get('pagaduria', 'N/A').upper()}")
        print(f"CLIENTE NUEVO:  {'SÍ' if historial['es_cliente_nuevo'] else 'NO'}")
        print("─"*60)
        print(f"PROB. APROBACIÓN: {res['prob_aprobacion']}%")
        print(f"COOPERATIVA SUGERIDA:  {res['cooperativa']}")
        print(f"CONFIANZA ASIGNACIÓN:  {res['confianza_coop']}%")
        print("─"*60)
        print(f"SCORE DE PRIORIDAD:    {res['score_prioridad']}%")
        print(f"CONTACTO CELULAR:      {datos.get('celular', 'No suministrado')}")
        print("═"*60 + "\n")
            
        return jsonify({
            'status': 'exitoso',
            'prospecto': datos.get('nombre'),
            'analisis': res
        }), 200

    except Exception as e:
        print(f"Error en el proceso: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)