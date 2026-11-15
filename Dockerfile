ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ENV TZ=Asia/Shanghai
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    tzdata \
    procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-portfolio-research.txt .
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.extra-index-url https://pypi.org/simple && \
    pip install --no-cache-dir -r requirements-portfolio-research.txt

COPY . .
COPY data-access-lib /app/data-access-lib
RUN pip install --no-build-isolation --no-deps /app/data-access-lib

RUN chmod +x /app/start.portfolio-researcher.sh

CMD ["/app/start.portfolio-researcher.sh"]
