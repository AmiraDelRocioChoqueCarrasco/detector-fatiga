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
- [ ] Métricas EAR/MAR y PERCLOS
- [ ] Partición por sujeto
- [ ] CNN de ojos
- [ ] Extracción de características de videos
- [ ] Modelo temporal
- [ ] Demo en tiempo real
- [ ] Pruebas

Voy a ir subiendo el avance por partes, no todo de golpe.

## Requisitos

Python 3.10, 3.11 o 3.12.
