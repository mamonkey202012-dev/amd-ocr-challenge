FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

# Salin dan pasang dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r /app/requirements.txt

# Salin script aplikasi
COPY app.py /app/app.py

# Buat folder output evaluasi
RUN mkdir -p /app/output

# Pre-download bobot model agar inferensi cepat saat dievaluasi
RUN python3 -c "from transformers import AutoProcessor, Qwen2VLForConditionalGeneration; AutoProcessor.from_pretrained('Qwen/Qwen2-VL-2B-Instruct'); Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')"

CMD ["/bin/bash"]
