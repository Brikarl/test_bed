"""TensorFlow Serving 客户端"""

import os
import pickle
from typing import Dict

import numpy as np
import requests


class TFServingClient:
    """TensorFlow Serving 客户端"""

    def __init__(self, host: str = "localhost", port: int = 8501):
        self.base_url = f"http://{host}:{port}"
        self.model_name = "wide_deep"
        self.model_version = "1"

        # 加载预处理器
        self.preprocessors = self._load_preprocessors()

    def _load_preprocessors(self):
        """加载预处理器"""
        try:
            preprocessor_path = os.path.join("models", "serving", "wide_deep", "preprocessors.pkl")
            if os.path.exists(preprocessor_path):
                with open(preprocessor_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"⚠️ 加载预处理器失败: {e}")
        return None

    def predict(self, features: Dict[str, np.ndarray]) -> float:
        """调用 TensorFlow Serving 进行预测"""
        try:
            # 构建请求 URL
            url = f"{self.base_url}/v1/models/{self.model_name}:predict"

            # 准备请求数据
            data = {
                "signature_name": "serving_default",
                "inputs": {
                    "wide": features['wide'].tolist(),
                    "deep": features['deep'].tolist(),
                    "query_hash": features['query_hash'].tolist(),
                    "doc_hash": features['doc_hash'].tolist(),
                    "position_group": features['position_group'].tolist()
                    }
                }

            # 发送请求
            response = requests.post(url, json=data)

            if response.status_code == 200:
                result = response.json()
                predictions = result['outputs']['ctr_score']
                return predictions[0][0] if isinstance(predictions, list) else predictions
            else:
                print(f"❌ TensorFlow Serving 返回错误: {response.status_code}")
                print(response.text)
                return 0.1

        except Exception as e:
            print(f"❌ 调用 TensorFlow Serving 失败: {e}")
            return 0.1

    def health_check(self) -> bool:
        """健康检查"""
        try:
            url = f"{self.base_url}/v1/models/{self.model_name}"
            response = requests.get(url)
            return response.status_code == 200
        except:
            return False
