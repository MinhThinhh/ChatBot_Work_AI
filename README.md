#Cách 1: 
cd đến thư mục chatbot_final chứa code
python -m venv venv
venv\Scripts\Activate
python.exe -m pip install --upgrade pip
pip install -r requirements.txt

streamlit run chat_app.py

#Cách 2: Chạy bằng Docker
- Mở terminal/command prompt
- Di chuyển đến thư mục chứa code
- Chạy lệnh:
  docker-compose up --build
- Mở trình duyệt và truy cập: http://localhost:8501

#cách 3: chạy trực tiếp từ docker image
- Mở terminal/command prompt
- Di chuyển đến thư mục chứa code là thư mục chatbot_final
- build image:
docker build -t ten_image .
docker run -d -p 8501:8501 -v "${PWD}:/app" -v "${PWD}/data:/app/data" -v "${PWD}/chroma_store:/app/chroma_store" -e OPENAI_API_KEY --gpus all --restart unless-stopped --name chatbot-container chatbot-image

- xem container: docker ps
- Mở trình duyệt và truy cập: http://localhost:8501
