FROM python_base
WORKDIR /app
ENV PYTHONPATH=/app/src PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BASELINE_CPU_THREADS=1
RUN --mount=type=bind,source=wheels,target=/wheels \
    pip install --no-cache-dir --no-index --find-links=/wheels -r /wheels/requirements.txt \
    && pip check
COPY src ./src
COPY harness.py contest_submission.py NOTICE.md SDK_SOURCE.json ./
COPY native/libgomp.so.1 /usr/lib/x86_64-linux-gnu/libgomp.so.1
USER 65534:65534
RUN ["python3", "-c", "import ctypes, json, lightgbm, scipy; from zhixun.c_boost_runtime import HEAD_KEY; ctypes.CDLL('libgomp.so.1'); assert lightgbm.__version__ == '4.6.0'; assert scipy.__version__ == '1.15.3'; assert HEAD_KEY == 'c_bounded_boost'; print(json.dumps({'lightgbm':lightgbm.__version__,'scipy':scipy.__version__,'openmp':'libgomp.so.1','c_head':HEAD_KEY}))"]
ENTRYPOINT ["python3", "/app/harness.py"]
CMD ["describe"]
COPY DEPENDENCY_ORIGINS.json ./
