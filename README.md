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
src/
  config.py                 rutas y parámetros
  features.py               EAR, MAR, PERCLOS y calibración
  splits.py                 partición por sujeto
  prepare_mrl.py            manifiesto de MRL Eye
  train_cnn.py              entrenamiento de la CNN de ojos
  extract_video_features.py EAR/MAR a partir de videos
  train_temporal.py         entrenamiento de la GRU
  realtime_demo.py          demo con webcam
tests/      pruebas (todavía no las subo)
```

## Estado

- [x] Estructura del repositorio
- [x] Métricas EAR/MAR y PERCLOS
- [x] Partición por sujeto
- [x] CNN de ojos
- [x] Extracción de características de videos
- [x] Modelo temporal
- [x] Demo en tiempo real
- [ ] Pruebas

La parte de pruebas la dejo para el final, por eso todavía no está en el repo.

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

## Entrenar la CNN de ojos

Primero tiene que existir el manifiesto de MRL. Después:

```bash
python src/train_cnn.py
```

Uso MobileNetV2 con transfer learning en dos fases:

1. Congelo el backbone y entreno solo la capa final (más rápido y evito
   romper los pesos preentrenados).
2. Descongelo las últimas 30 capas y hago un ajuste fino con learning rate
   mucho más bajo. Las capas de BatchNorm las mantengo congeladas.

Como en MRL hay más ojos abiertos que cerrados, uso `class_weight` para que el
modelo no se vaya hacia la clase mayoritaria. La evaluación final es sobre los
sujetos de test, que no se usaron en ninguna fase del entrenamiento.

Archivos que genera:

- `models/eye_cnn.keras`: el modelo.
- `models/eye_cnn.tflite`: solo si corro con `--export-tflite` (cuantización
  dinámica, para probar en algo más liviano).

La primera ejecución puede descargar los pesos de ImageNet. El umbral que uso
para reportar es 0.5 y las métricas dependen del dataset, así que no
significan que funcione igual en una webcam real.

## Videos y características

Para la parte temporal pensé en videos tipo YawDD, UTA-RLDD o NTHU, pero no
los incluyo en el repo. Los videos van en `data/raw/videos/`.

El problema es que cada dataset ordena y nombra los videos a su manera, así
que las funciones `infer_subject()` e `infer_label()` de
`src/extract_video_features.py` son solo un ejemplo. Hay que revisar la
estructura real antes de correrlo:

- `infer_subject()` tiene que devolver el mismo identificador para todos los
  videos de una misma persona (no uno por video).
- `infer_label()` tiene que devolver 0 para alerta y 1 para fatiga. La
  heurística que dejé busca "drowsy" o "yawn" en el nombre, pero eso no es una
  anotación confiable.

Después:

```bash
python src/extract_video_features.py
```

Se genera `data/processed/video_features.csv` con una fila por fotograma donde
MediaPipe detectó un rostro. Este CSV tampoco se sube (es generado y puede
tener datos de personas). Si un video no tiene detecciones, puede quedar con
muy pocas filas o vacío, así que conviene revisarlo antes de entrenar.

Uso `mediapipe<1.0` porque en la versión 1.x ya no está `mp.solutions`, que es
lo que necesito para Face Mesh.

## Modelo temporal

Con el CSV de características:

```bash
python src/train_temporal.py
```

Antes de armar las ventanas normalizo el EAR de cada video dividiéndolo por su
mediana, y agrego `ear_delta` (la diferencia entre fotogramas). La idea es que
el modelo no dependa del tamaño de ojo de cada persona.

Las ventanas son de 3 segundos con avance de 1 segundo, suponiendo 30 FPS
(todo eso se cambia en `src/config.py`). La etiqueta de cada ventana es la
mayoría de los fotogramas. Entreno una GRU chica (32 unidades) y vuelvo a
partir por sujeto, esta vez a nivel de ventana.

El modelo se guarda en `models/temporal_gru.keras`. Ojo: la demo de la webcam
no usa este modelo, usa EAR/MAR/PERCLOS y opcionalmente la CNN de ojos. La GRU
quedó como experimento con los videos.

## Demo en tiempo real

Esta es la parte que más me interesaba. Funciona con la webcam, OpenCV y
MediaPipe.

Solo con geometría (no necesita modelos entrenados):

```bash
python src/realtime_demo.py
```

Con la CNN entrenada:

```bash
python src/realtime_demo.py --cnn models/eye_cnn.keras
```

Si la cámara por defecto no es la 0, se puede cambiar con `--camera 1`.

Al inicio hace una calibración de 10 segundos: hay que mirar a la cámara con
los ojos abiertos y en condiciones normales. Con eso saca el umbral de EAR de
la persona. `r` vuelve a calibrar y `q` sale.

En pantalla se ven el EAR, el umbral, el MAR y el PERCLOS. Las alertas son:

- **MICROSUENO**: los ojos quedan debajo del umbral por más de 1 segundo
  seguido.
- **FATIGA**: el PERCLOS pasa del 15 %.
- **BOSTEZO**: el MAR pasa de 0.60 durante 1.5 segundos.
- Si uso `--cnn`, para marcar el ojo como cerrado pido que coincidan el EAR y
  la CNN (probabilidad mayor a 0.5), así no salta tan fácil por un parpadeo.

## Requisitos e instalación

- Python 3.10, 3.11 o 3.12.
- Webcam para la demo.
- Los datasets no vienen incluidos.

En Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

En Linux o macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si PowerShell no deja activar el entorno, se puede usar directamente
`.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.

TensorFlow puede necesitar una instalación aparte si se quiere usar GPU. En
Windows lo normal es que corra en CPU.

## Limitaciones y pendientes

- Los umbrales de EAR, MAR y PERCLOS son valores iniciales. No están
  calibrados para todas las personas, cámaras ni condiciones de luz; hay que
  ajustarlos con datos reales.
- En los videos, la etiqueta y el sujeto dependen de cómo esté ordenado el
  dataset. Si eso está mal, las métricas no sirven.
- La extracción temporal depende de que MediaPipe detecte el rostro y del FPS
  configurado. Conviene revisar el CSV antes de entrenar.
- Todavía falta la parte de pruebas unitarias.
- No hay validación clínica ni automotriz. Es un prototipo académico.

