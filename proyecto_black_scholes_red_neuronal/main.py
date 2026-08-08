"""
PROYECTO: RED NEURONAL PARA ESTIMAR EL PRECIO DE OPCIONES FINANCIERAS

Este programa combina el modelo matematico de Black-Scholes con una red
neuronal de tipo perceptron multicapa (MLP). Primero genera ejemplos mediante
la formula de Black-Scholes y despues entrena la red neuronal para aproximar
el precio de una opcion europea de compra (call).

Autor: [Escribir nombre del estudiante]
Curso: Redes Neuronales
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# Semilla fija para obtener los mismos resultados cada vez que se ejecuta.
SEMILLA = 42

# Nombres de las cinco variables de entrada de la red neuronal.
VARIABLES_ENTRADA = [
    "precio_accion_S",
    "precio_ejercicio_K",
    "tiempo_T",
    "tasa_interes_r",
    "volatilidad_sigma",
]


def precio_black_scholes_call(S, K, T, r, sigma):
    """Calcula el precio de una opcion europea de compra con Black-Scholes.

    Los parametros pueden ser numeros individuales o arreglos de NumPy.

    S     : precio actual de la accion.
    K     : precio de ejercicio de la opcion.
    T     : tiempo hasta el vencimiento, expresado en anios.
    r     : tasa de interes anual en formato decimal.
    sigma : volatilidad anual en formato decimal.
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if np.any(S <= 0) or np.any(K <= 0):
        raise ValueError("S y K deben ser mayores que cero.")
    if np.any(T <= 0) or np.any(sigma <= 0):
        raise ValueError("T y sigma deben ser mayores que cero.")

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (
        sigma * np.sqrt(T)
    )
    d2 = d1 - sigma * np.sqrt(T)

    precio_call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return precio_call


def generar_dataset(cantidad: int, semilla: int = SEMILLA) -> pd.DataFrame:
    """Genera casos simulados y calcula para cada uno el valor Black-Scholes."""
    if cantidad < 1000:
        raise ValueError("Se recomienda generar al menos 1000 ejemplos.")

    rng = np.random.default_rng(semilla)

    # Rangos simulados razonables para un experimento academico.
    S = rng.uniform(50.0, 200.0, cantidad)
    K = rng.uniform(50.0, 200.0, cantidad)
    T = rng.uniform(0.05, 2.0, cantidad)
    r = rng.uniform(0.00, 0.10, cantidad)
    sigma = rng.uniform(0.10, 0.60, cantidad)

    precio = precio_black_scholes_call(S, K, T, r, sigma)

    return pd.DataFrame(
        {
            "precio_accion_S": S,
            "precio_ejercicio_K": K,
            "tiempo_T": T,
            "tasa_interes_r": r,
            "volatilidad_sigma": sigma,
            "precio_opcion_black_scholes": precio,
        }
    )


def entrenar_red(dataset: pd.DataFrame):
    """Divide los datos, normaliza las variables y entrena la red neuronal."""
    X = dataset[VARIABLES_ENTRADA].to_numpy()
    y = dataset[["precio_opcion_black_scholes"]].to_numpy()

    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEMILLA,
    )

    # La normalizacion evita que las variables de mayor magnitud dominen.
    escalador_X = StandardScaler()
    escalador_y = StandardScaler()

    X_entrenamiento_escalado = escalador_X.fit_transform(X_entrenamiento)
    X_prueba_escalado = escalador_X.transform(X_prueba)
    y_entrenamiento_escalado = escalador_y.fit_transform(y_entrenamiento).ravel()

    # Arquitectura: 5 entradas -> 128 -> 64 -> 32 -> 1 salida.
    # MLPRegressor es un perceptron multicapa para problemas de regresion.
    red_neuronal = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.00001,
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=SEMILLA,
        verbose=False,
    )

    red_neuronal.fit(X_entrenamiento_escalado, y_entrenamiento_escalado)

    prediccion_escalada = red_neuronal.predict(X_prueba_escalado).reshape(-1, 1)
    prediccion = escalador_y.inverse_transform(prediccion_escalada).ravel()

    # El precio teorico de una opcion no puede ser negativo.
    prediccion = np.maximum(prediccion, 0.0)

    return {
        "modelo": red_neuronal,
        "escalador_X": escalador_X,
        "escalador_y": escalador_y,
        "X_prueba": X_prueba,
        "y_prueba": y_prueba.ravel(),
        "prediccion": prediccion,
    }


def calcular_metricas(y_real: np.ndarray, y_predicha: np.ndarray) -> dict:
    """Calcula medidas que permiten evaluar la calidad de la red."""
    mae = mean_absolute_error(y_real, y_predicha)
    rmse = np.sqrt(mean_squared_error(y_real, y_predicha))
    r2 = r2_score(y_real, y_predicha)

    # Se calcula MAPE solo para opciones con valor >= 1; cerca de cero el
    # porcentaje puede crecer demasiado aunque el error monetario sea pequeno.
    mascara = y_real >= 1.0
    mape = np.mean(
        np.abs((y_real[mascara] - y_predicha[mascara]) / y_real[mascara])
    ) * 100

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def crear_graficas(resultado: dict, carpeta_salida: Path) -> Path:
    """Crea una figura con la convergencia, comparacion y distribucion del error."""
    modelo = resultado["modelo"]
    y_real = resultado["y_prueba"]
    y_predicha = resultado["prediccion"]
    errores = y_predicha - y_real

    plt.style.use("seaborn-v0_8-whitegrid")
    figura, ejes = plt.subplots(1, 3, figsize=(18, 5.3))

    # Grafica 1: reduccion de la perdida durante el entrenamiento.
    ejes[0].plot(
        range(1, len(modelo.loss_curve_) + 1),
        modelo.loss_curve_,
        color="#1565C0",
        linewidth=2,
    )
    ejes[0].set_title("Convergencia del entrenamiento")
    ejes[0].set_xlabel("Iteración")
    ejes[0].set_ylabel("Función de pérdida")

    # Grafica 2: valores de Black-Scholes frente a la red neuronal.
    rng = np.random.default_rng(SEMILLA)
    cantidad_puntos = min(1500, len(y_real))
    indices = rng.choice(len(y_real), size=cantidad_puntos, replace=False)
    limite = max(y_real.max(), y_predicha.max())
    ejes[1].scatter(
        y_real[indices],
        y_predicha[indices],
        s=14,
        alpha=0.55,
        color="#00897B",
        edgecolors="none",
    )
    ejes[1].plot([0, limite], [0, limite], "--", color="#D32F2F", linewidth=2)
    ejes[1].set_title("Valor real frente a predicción")
    ejes[1].set_xlabel("Precio con Black-Scholes")
    ejes[1].set_ylabel("Precio de la red neuronal")

    # Grafica 3: distribucion de los errores de prediccion.
    ejes[2].hist(errores, bins=45, color="#7E57C2", alpha=0.85)
    ejes[2].axvline(0, color="#D32F2F", linestyle="--", linewidth=2)
    ejes[2].set_title("Distribución del error")
    ejes[2].set_xlabel("Prediccion - valor real")
    ejes[2].set_ylabel("Frecuencia")

    figura.suptitle(
        "Evaluación de la red neuronal aplicada al modelo Black-Scholes",
        fontsize=15,
        fontweight="bold",
    )
    figura.tight_layout()

    ruta_grafica = carpeta_salida / "resultados_red_neuronal.png"
    figura.savefig(ruta_grafica, dpi=180, bbox_inches="tight")
    return ruta_grafica


def predecir_caso(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    resultado: dict,
) -> tuple[float, float, float]:
    """Compara Black-Scholes y la red neuronal para un nuevo caso."""
    valores = np.array([[S, K, T, r, sigma]], dtype=float)
    precio_exacto = float(precio_black_scholes_call(S, K, T, r, sigma))

    valores_escalados = resultado["escalador_X"].transform(valores)
    prediccion_escalada = resultado["modelo"].predict(valores_escalados).reshape(
        -1, 1
    )
    precio_red = float(
        resultado["escalador_y"].inverse_transform(prediccion_escalada)[0, 0]
    )
    precio_red = max(precio_red, 0.0)
    error_absoluto = abs(precio_exacto - precio_red)
    return precio_exacto, precio_red, error_absoluto


def solicitar_datos_usuario(resultado: dict) -> None:
    """Permite ingresar otro caso desde la terminal de VS Code."""
    print("\n--- PRUEBA CON DATOS DEL USUARIO ---")
    print("Para una tasa de 5 % escriba 0.05; para 20 % escriba 0.20.")
    try:
        S = float(input("Precio actual de la accion S: "))
        K = float(input("Precio de ejercicio K: "))
        T = float(input("Tiempo hasta vencimiento en anios T: "))
        r = float(input("Tasa de interes anual r: "))
        sigma = float(input("Volatilidad anual sigma: "))
        exacto, neuronal, error = predecir_caso(S, K, T, r, sigma, resultado)
        print(f"Precio Black-Scholes : {exacto:.4f}")
        print(f"Precio red neuronal  : {neuronal:.4f}")
        print(f"Error absoluto       : {error:.4f}")
    except ValueError as error:
        print(f"No se pudo realizar la prueba: {error}")


def guardar_resultados(
    dataset: pd.DataFrame,
    resultado: dict,
    metricas: dict,
    carpeta_salida: Path,
) -> None:
    """Guarda el conjunto de datos, las predicciones, metricas y el modelo."""
    dataset.to_csv(carpeta_salida / "dataset_black_scholes.csv", index=False)

    tabla_predicciones = pd.DataFrame(
        resultado["X_prueba"], columns=VARIABLES_ENTRADA
    )
    tabla_predicciones["precio_real"] = resultado["y_prueba"]
    tabla_predicciones["precio_predicho"] = resultado["prediccion"]
    tabla_predicciones["error_absoluto"] = np.abs(
        resultado["y_prueba"] - resultado["prediccion"]
    )
    tabla_predicciones.to_csv(
        carpeta_salida / "predicciones_prueba.csv", index=False
    )

    pd.DataFrame([metricas]).to_csv(carpeta_salida / "metricas.csv", index=False)

    paquete_modelo = {
        "modelo": resultado["modelo"],
        "escalador_X": resultado["escalador_X"],
        "escalador_y": resultado["escalador_y"],
        "variables_entrada": VARIABLES_ENTRADA,
    }
    joblib.dump(paquete_modelo, carpeta_salida / "modelo_entrenado.joblib")


def mostrar_resumen(metricas: dict, resultado: dict, cantidad: int) -> None:
    """Muestra en la terminal el resumen principal del experimento."""
    modelo = resultado["modelo"]
    print("\n============================================================")
    print(" RED NEURONAL APLICADA AL MODELO BLACK-SCHOLES")
    print("============================================================")
    print(f"Ejemplos generados          : {cantidad}")
    print("División de datos           : 80 % entrenamiento / 20 % prueba")
    print("Arquitectura                : 5 - 128 - 64 - 32 - 1")
    print("Activación / optimizador    : ReLU / Adam")
    print(f"Iteraciones realizadas      : {modelo.n_iter_}")
    print(f"MAE                         : {metricas['MAE']:.6f}")
    print(f"RMSE                        : {metricas['RMSE']:.6f}")
    print(f"R cuadrado (R2)             : {metricas['R2']:.6f}")
    print(f"MAPE para precios >= 1      : {metricas['MAPE']:.3f} %")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena una red neuronal para aproximar Black-Scholes."
    )
    parser.add_argument(
        "--muestras",
        type=int,
        default=25000,
        help="Cantidad de ejemplos simulados (valor predeterminado: 25000).",
    )
    parser.add_argument(
        "--interactivo",
        action="store_true",
        help="Solicita un caso adicional desde la terminal.",
    )
    parser.add_argument(
        "--sin-mostrar",
        action="store_true",
        help="Guarda la grafica sin abrir una ventana.",
    )
    args = parser.parse_args()

    carpeta_salida = Path(__file__).resolve().parent / "resultados"
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    print("Generando el conjunto de datos...")
    dataset = generar_dataset(args.muestras)

    print("Entrenando la red neuronal...")
    resultado = entrenar_red(dataset)
    metricas = calcular_metricas(resultado["y_prueba"], resultado["prediccion"])

    guardar_resultados(dataset, resultado, metricas, carpeta_salida)
    ruta_grafica = crear_graficas(resultado, carpeta_salida)
    mostrar_resumen(metricas, resultado, args.muestras)

    # Prueba fija y reproducible para explicar el programa en la exposicion.
    exacto, neuronal, error = predecir_caso(
        S=100.0,
        K=105.0,
        T=1.0,
        r=0.05,
        sigma=0.20,
        resultado=resultado,
    )
    print("\n--- EJEMPLO DE COMPROBACIÓN ---")
    print("S=100, K=105, T=1, r=0.05, sigma=0.20")
    print(f"Precio Black-Scholes : {exacto:.4f}")
    print(f"Precio red neuronal  : {neuronal:.4f}")
    print(f"Error absoluto       : {error:.4f}")
    print(f"\nArchivos guardados en: {carpeta_salida}")
    print(f"Grafica principal: {ruta_grafica.name}")

    if args.interactivo:
        solicitar_datos_usuario(resultado)

    if not args.sin_mostrar:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
