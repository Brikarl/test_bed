"""基于 TensorFlow Serving 的模型服务"""

from datetime import datetime
from typing import Dict, Any

import requests
from flask import Flask, request, jsonify

from .ctr_wide_deep_model import WideAndDeepCTRModel
from .tf_serving_client import TFServingClient


class TFModelService:
    """使用 TensorFlow Serving 的模型服务"""

    def __init__(self, tf_serving_host: str = "tf-serving", tf_serving_port: int = 8501):
        self.tf_client = TFServingClient(tf_serving_host, tf_serving_port)
        self.wide_deep_model = WideAndDeepCTRModel()
        self.flask_app = None
        self.api_running = False

    def predict_ctr(self, features: Dict[str, Any]) -> float:
        """预测 CTR"""
        try:
            # 构建样本数据
            sample_data = [{
                'query': features.get('query', ''),
                'doc_id': features.get('doc_id', ''),
                'position': features.get('position', 1),
                'score': features.get('score', 0.0),
                'summary': features.get('summary', ''),
                'clicked': 0,
                'timestamp': features.get('timestamp', datetime.now().isoformat())
                }]

            # 提取特征
            extracted_features, _ = self.wide_deep_model.extract_features(sample_data, is_training=False)

            if len(extracted_features) == 0:
                return features.get('score', 0.1)

            # 标准化特征
            if self.tf_client.preprocessors:
                wide_scaler = self.tf_client.preprocessors.get('wide_scaler')
                deep_scaler = self.tf_client.preprocessors.get('deep_scaler')

                if wide_scaler:
                    extracted_features['wide'] = wide_scaler.transform(extracted_features['wide'])
                if deep_scaler:
                    extracted_features['deep'] = deep_scaler.transform(extracted_features['deep'])

            # 调用 TensorFlow Serving
            ctr_prob = self.tf_client.predict(extracted_features)

            # 返回加权分数
            return float(features.get('score', 0.0) * (1 + ctr_prob))

        except Exception as e:
            print(f"❌ 预测失败: {e}")
            return features.get('score', 0.1)

    def start_api_server(self, host: str = "0.0.0.0", port: int = 8502):
        """启动 API 服务器"""
        try:
            self.flask_app = Flask(__name__)
            self._setup_api_routes()

            print(f"🚀 TF Model Service API 启动在 {host}:{port}")
            print("📋 可用接口:")
            print(f"   - 健康检查: http://localhost:{port}/health")
            print(f"   - 预测接口: http://localhost:{port}/v1/predict")
            print(f"   - TF Serving 状态: http://localhost:{port}/tf-serving/status")

            self.api_running = True
            self.flask_app.run(host=host, port=port, debug=False, threaded=True)

        except Exception as e:
            print(f"❌ 启动 API 服务器失败: {e}")
            return False

    def _setup_api_routes(self):
        """设置 API 路由"""

        @self.flask_app.route('/health', methods=['GET'])
        def health():
            """健康检查"""
            tf_serving_healthy = self.tf_client.health_check()
            return jsonify(
                {
                    "status": "healthy" if tf_serving_healthy else "degraded",
                    "tf_serving": "connected" if tf_serving_healthy else "disconnected"
                    }
                )

        @self.flask_app.route('/tf-serving/status', methods=['GET'])
        def tf_serving_status():
            """TensorFlow Serving 状态"""
            try:
                response = requests.get(f"{self.tf_client.base_url}/v1/models/wide_deep")
                if response.status_code == 200:
                    return jsonify(response.json())
            except:
                pass
            return jsonify({"error": "Cannot connect to TensorFlow Serving"}), 503

        @self.flask_app.route('/v1/predict', methods=['POST'])
        def predict():
            """预测接口"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({"error": "No JSON data provided"}), 400

                ctr_score = self.predict_ctr(data)
                return jsonify({"ctr_score": ctr_score})

            except Exception as e:
                return jsonify({"error": str(e)}), 500
