FROM node:20-alpine

WORKDIR /app
COPY package.json ./
RUN npm install
COPY src ./src
COPY public ./public

ENV PORT=8080
EXPOSE 8080
CMD ["npm", "run", "start"]
