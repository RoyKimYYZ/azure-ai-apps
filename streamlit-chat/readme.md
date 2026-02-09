

[https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps]

## To run  streamlit chatbot in local browser
streamlit run chatbot.py --server.port 8502

##Attaching Debugger to Streamlit app running in browser

streamlit app python file is running in subfolder /streamlitchat so need to be explicit in launch.json file

In launch.json file, add
<pre>```
{
      "name": "Python:Streamlit",
      "type": "debugpy",
      "request": "launch",
      "module": "streamlit",
      "args": [
          "run",
          "${file}",
          "--server.port",
          "8502"
      ],
      "cwd": "${workspaceFolder}/streamlit-chat", // Set the working directory to the subfolder
    }```</pre>

Ensure that VS Code is using the correct Python interpreter:

Open the Command Palette (Ctrl+Shift+P).
Search for and select Python: Select Interpreter.
Choose the Python interpreter from your virtual environment 
```(e.g., /home/rkadmin/azureai-chatapp/streamlit-chat/venv/bin/python)```