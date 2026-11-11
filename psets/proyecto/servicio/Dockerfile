# La imagen del servicio de fallas de SSD. Se construye desde psets/proyecto/,
# porque necesita modelado/ (las features) ademas de servicio/. Antes de
# construirla, exporta el modelo: python servicio/exportar_modelo.py
#
#     docker build -f servicio/Dockerfile -t servicio-ssd .
#     docker run -p 8000:8000 servicio-ssd
FROM python:3.11-slim
WORKDIR /app
COPY servicio/requirements.txt servicio/requirements.txt
RUN pip install --no-cache-dir -r servicio/requirements.txt
COPY modelado/pipeline_proyecto.py modelado/
COPY servicio/api.py servicio/modelo.joblib servicio/modelo.json servicio/
EXPOSE 8000

CMD ["uvicorn", "servicio.api:app", "--host", "0.0.0.0", "--port", "8000"]
