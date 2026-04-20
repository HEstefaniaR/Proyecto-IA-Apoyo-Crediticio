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
        
        target_col = 'ID_CLIENTE'
        if target_col not in df_bd.columns:
            print(f"Alerta: No se encontró la columna {target_col}. Columnas detectadas: {list(df_bd.columns)}")
            return {'dias_desde_ultimo_credito': 9999, 'num_creditos_totales': 0, 'es_cliente_nuevo': True}

        df_bd[target_col] = df_bd[target_col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        cedula_str = str(cedula).replace('.0', '').strip()
        
        cliente = df_bd[df_bd[target_col] == cedula_str]
        
        if cliente.empty:
            print(f"🔍 Buscando cédula: '{cedula_str}' -> No encontrada en BD.")
            return {'dias_desde_ultimo_credito': 9999, 'num_creditos_totales': 0, 'es_cliente_nuevo': True}
        
        num_creditos = len(cliente)
        
        col_fecha = next((c for c in ['FECHA', 'FECHA_CREDITO', 'FECHA_DESEMBOLSO'] if c in df_bd.columns), None)
        
        if col_fecha:
            ultima_fecha = pd.to_datetime(cliente[col_fecha]).max()
            dias_desde_ultimo = (datetime.now() - ultima_fecha).days
        else:
            dias_desde_ultimo = 999999

        print(f"¡Cliente encontrado! Cédula: {cedula_str} | Créditos: {num_creditos}")
        return {
            'dias_desde_ultimo_credito': int(dias_desde_ultimo),
            'num_creditos_totales': int(num_creditos),
            'es_cliente_nuevo': False
        }
        
    except Exception as e:
        print(f"Error en buscar_historial_cliente: {e}")
        return {'dias_desde_ultimo_credito': 9999, 'num_creditos_totales': 0, 'es_cliente_nuevo': True}

PAGADURIAS_PENSION = {'COLPENSIONES', 'ARP POSITIVA', 'FOPEP', 'FIDUPREVISORA', 'EJERCITO NACIONAL'}

def evaluar_reglas_negocio(edad, monto, plazo, tipo, pagaduria):
    # Simplificación de tus funciones de elegibilidad para la API
    resultados = {}
    
    # 1. Business Integrals
    razones_b = []
    if edad < 18 or edad > 84: razones_b.append(f"Edad {edad} fuera de rango (18-84)")
    if tipo != 'REFINANCIACION' and plazo < 49: razones_b.append(f"Plazo {plazo}m insuficiente para crédito nuevo")
    # ... (puedes copiar el resto de la lógica de _elegibilidad_business aquí)
    resultados['BUSINESS'] = {'ok': len(razones_b) == 0, 'razones': razones_b}

    # 2. ExcelCredit
    razones_e = []
    if edad < 20 or edad > 81: razones_e.append(f"Edad {edad} fuera de rango (20-81)")
    if monto < 1500000: razones_e.append("Monto menor al mínimo $1.5M")
    resultados['EXCELCREDIT'] = {'ok': len(razones_e) == 0, 'razones': razones_e}

    # 3. Coops Pensión (COPFINANCIAR, PRONALCREDIT, COOPIDESARROLLO)
    razones_p = []
    pag_ok = any(p in pagaduria.upper() for p in PAGADURIAS_PENSION)
    if not pag_ok: razones_p.append("Pagaduría no cubierta (requiere pensión pública)")
    if edad > 75: razones_p.append("Edad supera máximo de 75 años")
    
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

        # --- ETAPA 1: REGLAS DE NEGOCIO ---
        analisis_reglas = evaluar_reglas_negocio(edad, monto, plazo, tipo, pag)
        
        # --- ETAPA 2: MACHINE LEARNING ---
        # Creamos DataFrames para evitar los UserWarnings de sklearn
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
        
        # Modelo 1
        df1 = pd.DataFrame([[historial['dias_desde_ultimo_credito'], monto, plazo, historial['num_creditos_totales'], edad]],
                           columns=['dias_desde_ultimo_credito', 'MONTO', 'PLAZO', 'num_creditos_totales', 'EDAD_AL_PRESTAMO'])
        prob_aprob = m1.predict_proba(df1)[0][1]

        # Evitar el error '1' (KeyError) si ranking falla
        mejor_elegible = next((c for c in ranking if c['elegible_reglas']), None)
        if mejor_elegible is None and len(ranking) > 0:
            mejor_elegible = ranking[0]

        return {
            'prob_aprobacion': round(float(prob_aprob) * 100, 2),
            'ranking_cooperativas': ranking,
            'mejor_opcion_elegible': mejor_elegible
        }
    except Exception as e:
        print(f"Error dentro de predecir_oportunidad: {e}")
        raise e

# ===== ENDPOINTS =====

@app.route('/api/evaluar_desde_archivo', methods=['POST'])
def evaluar_desde_archivo():
    try:
        req = request.json
        rel_path = req.get('archivo_path').replace('./', '').replace('../', '')
        path_archivo = os.path.join(BASE_DIR, rel_path)
        
        with open(path_archivo, 'r') as f:
            lineas = f.readlines()
            datos = json.loads(lineas[-1]) 
            
        historial = buscar_historial_cliente(datos['cedula'])
        
        res = predecir_oportunidad(datos, historial)
        
        print("\n" + "═"*70)
        print("            FICHA TÉCNICA Y REGLAS DE NEGOCIO")
        print("═"*70)
        
        nombre = str(datos.get('nombre', 'N/A')).upper()
        cedula = datos.get('cedula', 'N/A')
        pagaduria = str(datos.get('pagaduria', 'N/A')).upper()
        edad = datos.get('edad', 'N/A')
        
        print(f"CLIENTE: {nombre} | CÉDULA: {cedula}")
        print(f"PAGADURÍA: {pagaduria} | EDAD: {edad}")
        
        # --- LÓGICA DE CLIENTE NUEVO O ANTIGUO ---
        if historial['es_cliente_nuevo']:
            print(f"ESTADO: CLIENTE NUEVO (Sin historial en BD)")
        else:
            print(f"ESTADO: CLIENTE ANTIGUO")
            print(f"   └─ Créditos totales: {historial['num_creditos_totales']}")
            print(f"   └─ Días desde último crédito: {historial['dias_desde_ultimo_credito']} días")
        
        print("─"*70)
        print(f"VIABILIDAD DE CRÉDITO (M1): {res['prob_aprobacion']}%")
        print("\nANÁLISIS POR COOPERATIVA (Ranking ML):")
        
        for i, item in enumerate(res['ranking_cooperativas'][:3], 1):
            status = "ELEGIBLE" if item['elegible_reglas'] else "NO ELEGIBLE"
            print(f"{i}. {item['cooperativa']:<18} | Confianza: {item['prob_ml']}% | {status}")
            if not item['elegible_reglas']:
                for razon in item['razones_rechazo']:
                    print(f"   └─ MOTIVO: {razon}")
        
        print("─"*70)
        print(f"CONTACTO: {datos.get('celular', 'N/A')}")
        print("═"*70 + "\n")
        
        res['historial_cliente'] = historial
        
        return jsonify({'status': 'procesado', 'resultados': res}), 200

    except Exception as e:
        print(f"Error en el proceso: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)