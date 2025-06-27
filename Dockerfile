# 使用官方 Node.js 映像檔
FROM node:18-alpine

# 設定工作目錄
WORKDIR /app

# 複製根目錄的 package.json（如果存在）
COPY package*.json ./

# 複製前端專案並建置
COPY frontend/ ./frontend/
WORKDIR /app/frontend
RUN npm install
RUN npm run build

# 回到根目錄，複製後端檔案
WORKDIR /app
COPY backend/ ./backend/

# 安裝後端依賴
WORKDIR /app/backend
RUN npm install --production

# 將前端建置檔案移到後端 public 目錄
RUN mkdir -p public
RUN cp -r ../frontend/build/* public/

# 暴露端口
EXPOSE 3001

# 啟動應用
CMD ["npm", "start"] 