# DriftGuard 评测 Harness

本仓库保存 DriftGuard A 气体识别、B 轴承寿命和 C 发动机寿命试运行镜像对应的源码。每轮使用平台授权的训练与公开验证投影，保留父检查点校验、候选审计和模型选择记录。

## 镜像构建

使用 Python 3.12 Linux/amd64 OCI 基础镜像作为命名构建上下文 `python_base`。根据 `DEPENDENCY_ORIGINS.json` 的摘要备齐 `wheels/` 和 `native/libgomp.so.1`。依赖包括通过赛事渠道取得的 SDK wheel。按 `Dockerfile` 使用 Docker Buildx 装配。默认入口为 `python3 /app/harness.py describe`，宿主按工作区协议提供训练、预测和 Sampler 网关。

## 来源与复核

`RELEASE_MANIFEST.json` 把镜像归档 SHA-256 与源码摘要绑定。`SDK_SOURCE.json`、`DEPENDENCY_ORIGINS.json` 和 `NOTICE.md` 记录软件和数据的来源。比赛数据、权重和账户密钥通过各自的授权渠道提供。
