# Detector de fatiga y somnolencia

Proyecto del curso de Inteligencia Artificial (Sexto Semestre).

La idea es armar un prototipo que detecte señales de fatiga usando la webcam:
ojos cerrados, microsueños, bostezos y cosas así.

**Autora:** Amira Del Rocío Choque Carrasco

## De qué se trata

Quiero medir algunas señales que se pueden sacar de la cara sin mucho lío:

- **EAR** (Eye Aspect Ratio): qué tanto se abre o se cierra el ojo.
- **MAR** (Mouth Aspect Ratio): qué tanto se abre la boca, para los bostezos.
- **PERCLOS**: porcentaje de tiempo con los ojos cerrados dentro de una
  ventana de tiempo.

Después la idea es complementar con una CNN que diga si el ojo está abierto o
cerrado, y una GRU que mire varios segundos seguidos de esas mediciones. Al
final quiero una demo en tiempo real con OpenCV y MediaPipe.

## Aviso

Esto es solo un trabajo académico. No es un dispositivo médico ni sirve para
decidir si alguien puede conducir. No hay que probar la demo manejando.

## Estructura

```text
data/       datos locales (no se suben al repo)
models/     modelos entrenados (no se suben al repo)
src/        código del proyecto
tests/      pruebas (todavía no las subo)
```

## Estado

- [x] Estructura del repositorio
- [x] Métricas EAR/MAR y PERCLOS
- [x] Partición por sujeto
- [ ] CNN de ojos
- [ ] Extracción de características de videos
- [ ] Modelo temporal
- [ ] Demo en tiempo real
- [ ] Pruebas

Voy a ir subiendo el avance por partes, no todo de golpe.

## Cómo calculo las métricas

En `src/features.py` están las funciones que uso en todo el proyecto. La idea
es que dependan solo de numpy para poder revisarlas rápido:

- **EAR**: tomo seis puntos de cada ojo (los índices del Face Mesh de
  MediaPipe) y divido la suma de las distancias verticales entre dos veces la
  distancia horizontal. Si el ojo está abierto el valor sube; si se cierra,
  baja.
- **MAR**: hago algo parecido con la boca, con tres aperturas verticales y el
  ancho entre las comisuras. Sirve para ver bostezos.
- **PERCLOS**: guardo muestras de tiempo con el estado del ojo y calculo qué
  porcentaje del tiempo estuvo cerrado en la ventana (por defecto 60 s).
- **Calibración**: en vez de usar un umbral fijo para todos, saco el umbral
  como una fracción del EAR mediano de la persona con los ojos abiertos.

Todo lo que son rutas e hiperparámetros está en `src/config.py`.

## Datos y partición

Para los ojos voy a usar el dataset **MRL Eye**. Hay que descargarlo aparte y
descomprimirlo en `data/raw/mrl_eye/`. Los archivos vienen con nombres como
`s0001_00001_0_0_0_0_0_01.png`, así que de ahí saco el sujeto, si usa gafas y
si el ojo está cerrado o abierto.

Con `python src/prepare_mrl.py` genero `data/processed/mrl_manifest.csv`, que
tiene la ruta de cada imagen, el sujeto, la etiqueta y el split. El CSV queda
ignorado por Git porque es un archivo generado.

La partición es 70/15/15 (train/val/test) y la hago **por sujeto** en
`src/splits.py`, no por imagen. Si parto las imágenes al azar, la misma persona
puede aparecer en train y en test y las métricas ya no reflejan nada real.
Para que sea reproducible uso la semilla 42.

## Requisitos

Python 3.10, 3.11 o 3.12.
