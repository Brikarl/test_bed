# API 服务镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制源代码
COPY src/ ./src/
COPY models/serving/wide_deep/preprocessors.pkl ./models/serving/wide_deep/

# 复制启动脚本
COPY run_tf_model_api.py .

# 设置环境变量
ENV PYTHONPATH=/app

# 暴露端口
EXPOSE 8502

# 运行 API 服务
CMD ["python", "run_tf_model_api.py"]
