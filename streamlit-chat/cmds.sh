
which python3
python3 --version

# create virutal environment
python3 -m venv venv
# activate virtual environment
source venv/bin/activate
deactivate

# https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps

pip install --upgrade pip
pip install -r requirements.txt

python config.py

streamlit run chatbot.py
streamlit run chatbot-agentic.py

pip install debugpy
python -m debugpy --listen 8502  --wait-for-client chatbot-agentic.py

pip install prompt-toolkit==3.0.43
pip install "prompty[azure]"
pip uninstall "prompty"

streamlit run chatbot-agentic.py


docker build -t streamlit-chatbot .

docker run -p 8501:8501 streamlit-chatbot

# list docker images
docker images
# list docker containers
docker ps -a
# remove docker container
docker rm 3af379ad8303
# remove docker image
docker rmi ce8f7398433c
# remove all docker containers