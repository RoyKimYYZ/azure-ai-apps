from flask import Flask, request, jsonify

from chat_with_resumes import chat_with_resumes  # Assuming this function exists in chat_with_resumes.py

app = Flask(__name__)
app.debug = True
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

@app.route('/api/chat', methods=['POST'])
def chat():
    # try:
        # Get the input data from the request
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'[/api/chat] error': 'Invalid input, "query" is required'}), 400

        query = data['query']
        print(f"[/api/chat] Received query: **************************************")
        print(query)
        # Call the chat_with_resumes function in chat_with_resumes.py
        response = chat_with_resumes([{"role": "user", "content": query}])
        print(f"[/api/chat]  Response from chat_with_resumes: ********************")
        print(response.choices[0].message) # { 'content': ''}
        
         # Convert `ChatCompletions` to a dictionary or string
        if hasattr(response.choices[0].message, 'to_dict'):  # Example: Check if it has a `to_dict` method
            print(f"[/api/chat]  ChatCompletions response: {type(response.choices[0].message)}")
            return jsonify(response.choices[0].message.to_dict())
        elif isinstance(response.choices[0].message, str):  # If it's already a string
            print(f"[/api/chat] String response: {type(response.choices[0].message)}")
            return jsonify(response.choices[0].message)
        else:
            # Handle other cases (e.g., extract relevant data, ChatResponse object)
            # print(f"[/api/chat] Unknown response type: {type(response.choices[0].message)}")
            # This resolves issue where response.choices[0].message is a ChatResponse object
            # and we need to extract the content and role. jsonify(response.choices[0].message) does not work
            # and raises TypeError: Object of type ChatResponse is not JSON serializable
            # Assuming `message` is an object with `content` and `role` attributes
            message = response.choices[0].message
            serialized_message = {
                "content": getattr(message, "content", None),
                "role": getattr(message, "role", None),
            }
            return jsonify(serialized_message)
    
        # Return the response as JSON
        return response, 200
    # TODO: how to deal with this error? for now, comment out exception handling
    # [chat_with_resumes] An unexpected error occurred: 'AIProjectClient' object has no attribute 'exceptions'
    # /api/chat > Error: 'AIProjectClient' object has no attribute 'exceptions'
    # except Exception as e:
    #     print(f"/api/chat > Error: {e}")
    #     return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)