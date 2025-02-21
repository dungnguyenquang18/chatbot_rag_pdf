from flask import Flask, request, jsonify
from demochatbot import Chatbot
app = Flask(__name__)
chatbot = Chatbot()

@app.route('/api/chatbot', methods=['POST'])
def handle_query():
    data = request.get_json()
    question = data.get('query')
    if not question:
        return jsonify({'error': 'No query provided'}), 400
    answer = chatbot.answer(question)
    # return answer
    return jsonify(answer)

if __name__ == '__main__':
    app.run(debug=True)
