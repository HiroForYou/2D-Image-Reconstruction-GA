# Reconstrucción de imágenes con algoritmos genéticos
## _Script para reconstruir imágenes en blanco y negro mediante la colocación aleatoria de formas_

## Instalación
```sh
pip install -r requirements.txt
```

## Funciones/parámetros clave del algoritmo genético

### Generación / población
Una generación consta de n (predeterminado: 50) imágenes en blanco y negro que contienen varias formas/textos

### Cruce
Varias funciones para cruce:
- blending (con canal alfa 0.5) (recomendado)
- Intercambio aleatorio de filas/columnas
- Concatenar dos mitades juntas

### Función de fitness
Relación señal/ruido máxima (PSNR) utilizada como función para evaluar la similitud de dos imágenes

### Mutación
Mute agregando un número de forma/texto aleatorio a la imagen