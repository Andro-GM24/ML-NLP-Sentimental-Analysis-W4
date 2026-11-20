# Evaluación de reconstrucción y anomalías

## Ejecución

Desde la raíz del repositorio:

```bash
python3 src/evaluate_ae.py --check-only
.venv/bin/python src/evaluate_ae.py
```

La primera orden revisa los archivos sin requerir dependencias de machine learning.
La segunda requiere el entorno del proyecto con PyTorch, NumPy, scikit-learn,
Matplotlib y joblib. No entrena ni modifica los checkpoints o el normalizador.

Estado de entradas: modelos, etiquetas, splits, normalizador y embeddings disponibles.
Los embeddings se descargaron de Git LFS y se verificaron por tamaño, SHA256,
forma (42632, 768) y valores finitos. La evaluación usa exclusivamente los
embeddings de `data/gold`; el aumento de datos es un experimento separado.

## Protocolo implementado

- Se verifica el hash del split original, la forma de los arrays, las etiquetas,
  la ausencia de solapamientos entre splits y los valores finitos.
- Se utiliza el normalizador guardado; no se vuelve a ajustar.
- Se evalúan AE y VAE con beta 0.5, 1 y 4 en modo evaluación, sin gradientes.
- La reconstrucción VAE usa la media latente, sin muestreo aleatorio.
- Se calcula MSE y correlación Pearson por reseña en el espacio normalizado.
  Las correlaciones indefinidas se excluyen del promedio y se contabilizan.
- Se resumen métricas globales y por clase. Los positivos de prueba permiten
  examinar reconstrucción de datos de la misma clase usada en entrenamiento;
  el test mixto mide además la reconstrucción de las clases no entrenadas.
- Para los paneles se selecciona el VAE de menor MSE global de validación.
  Se muestran nueve reseñas: percentiles aproximados 10, 50 y 90 del MSE del
  AE dentro de cada clase de prueba. Los mismos ejemplos se usan en ambos modelos.
  Las bandas muestran componentes del embedding, no imágenes ni texto reconstruido.
- Normal = positivo; anomalía = negativo o neutral.
- Scores: MSE del AE; MSE y MSE + KL de cada VAE. KL se promedia sobre las
  dimensiones latentes, en concordancia con la implementación actual de la pérdida.
- Cada umbral es el percentil 95 de scores positivos de validación. Se predice
  anomalía cuando score > umbral. No se calibra usando prueba.
- AUC-ROC se calcula con scores continuos sobre el test mixto. Precisión, recall,
  F1 y matriz de confusión usan el umbral fijado en validación.

## Archivos producidos al ejecutar con embeddings reales

- `runs/workshop4/evaluation/input_status.json`: disponibilidad de entradas.
- `runs/workshop4/evaluation/metrics.json`: protocolo y métricas de reconstrucción y anomalías.
- `runs/workshop4/evaluation/test_scores.csv`: resultados por reseña y modelo.
- `notebooks/figures/evaluation/reconstruction_panels.png`: original, AE, VAE y diferencias absolutas.
- `notebooks/figures/evaluation/reconstruction_by_class.png`: MSE por clase.
- `notebooks/figures/evaluation/anomaly_roc.png`: comparación ROC de siete detectores.
- `notebooks/figures/evaluation/anomaly_scores.png`: distribuciones de scores y umbrales.

## Redacción del informe

La evaluación ya se ejecutó. Los resultados y el análisis están en
[evaluacion_resultados.md](evaluacion_resultados.md). Las siguientes indicaciones
se conservan como guía metodológica.

### Calidad de reconstrucción

Evaluamos embeddings de 768 dimensiones normalizados con el scaler del entrenamiento.
El AE y los VAE comprimen estos vectores y reconstruyen su representación numérica.
Reportamos MSE y correlación de Pearson, globalmente y por sentimiento.

Insertar tabla de los cuatro modelos con MSE y Pearson. Discutir el compromiso entre
reconstrucción y regularización para cada beta. No comparar directamente las pérdidas
MSE + beta KL como si fueran una métrica de reconstrucción común.

Insertar los paneles y boxplots. Describir qué clases y ejemplos tienen errores mayores,
qué solapamientos aparecen y si existen positivos difíciles de reconstruir. Para atribuir
los errores a ironía, longitud o sentimientos mezclados se necesita consultar el texto
original; esas causas no se deducen únicamente de los embeddings.

### Detección de anomalías

Definimos las reseñas positivas como normales y las neutrales y negativas como anomalías.
Utilizamos modelos que se espera hayan sido entrenados únicamente con positivos mediante
`train_ae.py`. Verificar esta procedencia con el equipo: los pesos por sí solos no prueban
qué datos se utilizaron. No utilizar `vae_all_classes.pt` para este experimento.

Insertar tabla con AUC-ROC, umbral, precisión, recall y F1 de los siete detectores.
Insertar las curvas ROC y distribuciones. Interpretar los falsos positivos y negativos
usando las matrices guardadas en el JSON, con filas reales y columnas predichas en orden
[normal, anomalía]. Un buen error promedio de reconstrucción no garantiza buen AUC:
importa separar los scores de normales y anomalías.

### Comparación con W1 y W2

Completada en `evaluacion_resultados.md` y `notebooks/evaluation_ae.ipynb`.
W1 se reprodujo con el dataset recuperado y selección por macro-F1 de validación;
W2 se evaluó desde probabilidades guardadas, verificando identificadores y etiquetas.
Se utiliza `1 - P(positivo)` y AUC binario sobre el mismo test. Los scripts de
reproducción son `src/compare_w1.py`, `src/compare_w2.py` y
`src/plot_baseline_comparison.py`. Para W1 necesitan el código del repositorio vecino;
para W2, su CSV de probabilidades. Cada script acepta rutas alternativas.

Los textos originales ya están disponibles en `data/silver/reviews.parquet`;
el notebook incorpora fragmentos de los nueve ejemplos de reconstrucción.

## Precauciones del entrenamiento existente

El script de entrenamiento selecciona checkpoints con validación mixta aunque entrena
solo con positivos. Además, la selección VAE compara pérdidas mientras cambia el peso
KL durante annealing. La evaluación no corrige retrospectivamente esas decisiones;
deben documentarse o revisarse con quienes entrenaron antes de conclusiones definitivas.
