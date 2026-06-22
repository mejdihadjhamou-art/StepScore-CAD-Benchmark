FROM node:20-bookworm-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
  python3 \
  python3-pip \
  python3-numpy \
  python3-scipy \
  && rm -rf /var/lib/apt/lists/*

COPY services/worker/package.json ./package.json
RUN npm install
RUN pip3 install --no-cache-dir --break-system-packages trimesh

COPY services/worker/src ./src
COPY tools ./tools

CMD ["npm", "run", "start"]
