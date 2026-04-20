import joblib
import pandas as pd

# Rutas de tus modelos
m1_path = '/Users/estefania/Documents/proyecto IA Apoyo Crediticio/modelos/modelo1_random_forest.pkl'
m2_path = '/Users/estefania/Documents/proyecto IA Apoyo Crediticio/modelos/modelo2_Gradient_Boosting.pkl'

def inspeccionar(path, nombre_modelo):
    print(f"\n--- Analizando {nombre_modelo} ---")
    obj = joblib.load(path)
    
    # Si es un diccionario, buscamos el modelo real dentro
    modelo = None
    if isinstance(obj, dict):
        print(f"El archivo es un diccionario. Claves encontradas: {list(obj.keys())}")
        for k, v in obj.items():
            if hasattr(v, 'predict'):
                modelo = v
                print(f"Modelo encontrado en la clave: '{k}'")
                break
    else:
        modelo = obj
        print("El archivo es el objeto de modelo directo.")

    # Intentar sacar los nombres de las columnas (Features)
    if modelo is not None:
        if hasattr(modelo, 'feature_names_in_'):
            print(f"ORDEN DE COLUMNAS ESPERADO:")
            for i, col in enumerate(modelo.feature_names_in_, 1):
                print(f"{i}. {col}")
        else:
            print("El modelo no tiene grabado el nombre de las columnas (feature_names_in_).")
    else:
        print("No se pudo extraer un modelo válido del archivo.")

inspeccionar(m1_path, "MODELO 1")
inspeccionar(m2_path, "MODELO 2")