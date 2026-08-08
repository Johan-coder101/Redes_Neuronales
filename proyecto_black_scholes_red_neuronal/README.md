# Red neuronal aplicada a Black-Scholes

Este proyecto se realizó tomando como referencia el video **La ecuación del billón de dólares**. En el video se explica cómo el modelo Black-Scholes permitió calcular el valor aproximado de las opciones financieras.

La idea del programa es comprobar si una red neuronal puede aprender a obtener resultados cercanos a los de esa fórmula. Para realizar la prueba se generaron diferentes combinaciones de precios, tasas de interés, tiempos y volatilidades.

## ¿Qué datos utiliza?

La red neuronal trabaja con cinco valores:

* `S`: precio actual de la acción.
* `K`: precio de ejercicio de la opción.
* `T`: tiempo que falta para el vencimiento.
* `r`: tasa de interés anual.
* `sigma`: volatilidad del precio de la acción.

Con esos datos, la fórmula Black-Scholes calcula el precio teórico de una opción de compra. Ese resultado se utiliza como respuesta correcta durante el entrenamiento de la red neuronal.

## ¿Cómo funciona el programa?

Primero se generan 25 000 ejemplos utilizando valores aleatorios dentro de rangos establecidos. Después, la fórmula Black-Scholes calcula el precio correspondiente a cada ejemplo.

Los datos se dividen de la siguiente manera:

* 80 % para entrenar la red neuronal.
* 20 % para comprobar su funcionamiento.

Antes del entrenamiento se normalizan los datos, porque algunas variables tienen valores grandes y otras se expresan como decimales.

La red utilizada tiene cinco entradas, tres capas ocultas de 128, 64 y 32 neuronas, y una salida. Se utilizó la función de activación ReLU y el optimizador Adam.

## Archivos del proyecto

El proyecto contiene los siguientes archivos:

* `main.py`: contiene todo el código del programa.
* `requirements.txt`: contiene las librerías necesarias.
* `README.md`: contiene esta explicación.
* `resultados/`: se crea después de ejecutar el programa.

Dentro de la carpeta `resultados` se guardan el conjunto de datos, las predicciones, las métricas, las gráficas y el modelo entrenado.

## ¿Cómo ejecutarlo?

Primero se debe abrir la carpeta del proyecto en Visual Studio Code. Después, en la terminal, se instalan las librerías:

```powershell
python -m pip install -r requirements.txt
```

Luego se ejecuta el programa:

```powershell
python main.py
```

Si se desea ingresar valores propios desde la terminal, se utiliza:

```powershell
python main.py --interactivo
```

Las tasas deben escribirse en forma decimal. Por ejemplo, una tasa de interés del 5 % se escribe como `0.05` y una volatilidad del 20 % como `0.20`.

## Resultados de la prueba

En la ejecución realizada se obtuvieron estos resultados:

* MAE: 0.340869
* RMSE: 0.468416
* R²: 0.999825
* Iteraciones realizadas: 27

También se probó el siguiente caso:

```text
S = 100
K = 105
T = 1
r = 0.05
sigma = 0.20
```

El precio obtenido mediante Black-Scholes fue `8.0214`, mientras que la red neuronal calculó `8.1071`. La diferencia entre los dos resultados fue `0.0857`.

Estos resultados muestran que la red logró aproximarse correctamente a la fórmula dentro de los rangos utilizados para generar los datos.

## Aclaración

Este programa no intenta predecir si una acción subirá o bajará. Su finalidad es académica y consiste en demostrar cómo una red neuronal puede aprender a aproximar el resultado de una fórmula matemática.
