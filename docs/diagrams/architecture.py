"""README용 아키텍처 그림 생성 스크립트.

실행: pip install diagrams (graphviz 필요) 후 저장소 루트에서
    python docs/diagrams/architecture.py
결과: docs/images/architecture.png
"""
from diagrams import Cluster, Diagram, Edge
from diagrams.generic.storage import Storage
from diagrams.onprem.client import Client, User
from diagrams.onprem.container import Docker
from diagrams.onprem.mlops import Mlflow
from diagrams.programming.framework import FastAPI
from diagrams.programming.language import Python

FONT = "Apple SD Gothic Neo"
graph_attr = {"fontname": FONT, "fontsize": "15", "pad": "0.4", "nodesep": "0.7", "ranksep": "1.0", "bgcolor": "white"}
node_attr = {"fontname": FONT, "fontsize": "12"}
edge_attr = {"fontname": FONT, "fontsize": "11"}

with Diagram("", filename="docs/images/architecture", outformat="png", show=False, direction="LR",
             graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr):
    ops = User("공항 운영 담당자")
    dash = Client("운영 대시보드\n(static/index.html)")

    with Cluster("Docker 컨테이너 1대 (serving_app)", graph_attr={"fontname": FONT, "bgcolor": "#F4F7FB"}):
        Docker("python:3.11-slim")
        api = FastAPI("FastAPI\n/predict /data /monitor /logs")
        with Cluster("AIOps 루프", graph_attr={"fontname": FONT}):
            detector = Python("drift_detector\n이상치 / bias / 구조")
            trigger = Python("retrain_trigger\nfine-tune 3 epoch\n게이트 RMSE 2,700")
        model = Python("LSTM 3층\n(TensorFlow/Keras)\n_model_cache")
        registry = Mlflow("MLflow Registry\nsqlite mlflow.db\nProduction 단계")
        uploads = Storage("data/uploads\n(최신 CSV)")
        logs = Storage("logs/aiops.log")

    ops >> dash
    dash >> Edge(label="REST (CSV 업로드, 배치, 예측)") >> api
    dash << Edge(label="5초 폴링 (버전, 로그)", style="dashed") << api
    api >> Edge(label="업로드 저장") >> uploads
    api >> Edge(label="예측") >> model
    model << Edge(label="Production 로드\n(run_id, 번호)") << registry
    api >> Edge(label="예측 vs 실제 21건") >> detector
    detector >> Edge(label="|bias| > 500") >> trigger
    uploads >> Edge(label="최근 41행") >> trigger
    trigger >> Edge(label="등록, 승격, 롤백") >> registry
    trigger >> Edge(label="reset_cache()", style="dashed") >> model
    trigger >> Edge(label="[WARN] [INFO] [OK]") >> logs
    api >> Edge(label="stale 비교", style="dotted") >> registry
