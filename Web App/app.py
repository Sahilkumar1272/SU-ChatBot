
from flask import Flask, jsonify, render_template, request
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import re
import json
from groq import Groq
import time
import csv
import psycopg2


app = Flask(__name__)


grok_api_key = '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'
# Load model once during the server startup
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
messages = []  # Initialize global message list


def get_db_connection():
    # Connect to your PostgreSQL database
    conn = psycopg2.connect(
        dbname="chatbot",
        user="postgres",
        password="1234",
        host="localhost",  # Or the IP address of your PostgreSQL server
        port="5432"        # Default port for PostgreSQL
    )
    return conn

def get_top_answer(query, model, conn, top_k=10):
    # Generate embedding for the input query
    query_embedding = model.encode(query, convert_to_tensor=False)
    query_embedding = np.array(query_embedding, dtype='float32')

    # Convert query_embedding to string format suitable for PostgreSQL
    embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

    cursor = conn.cursor()

    # Perform similarity search using pgvector's <=> operator for cosine similarity
    cursor.execute("""
    SELECT id, question, answer, embedding
    FROM questions
    ORDER BY embedding <=> %s
    LIMIT %s
    """, (embedding_str, top_k))

    rows = cursor.fetchall()

    retrieved_docs = []
    for row in rows:
        retrieved_docs.append(row[2])  # Answer column

    cursor.close()

    return retrieved_docs

def GroqChat(question):
    client = Groq(
        api_key=grok_api_key,

    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ],
            model = "llama-3.1-70b-versatile"
    )

    cleaned_json_string = chat_completion.choices[0].message.content

    json_str = re.sub(r'}\s*{', '}, {', cleaned_json_string)
    return json_str

def generate_answer_from_docs(query, retrieved_docs):
    result = []
    if not retrieved_docs:
        return "Don't have an answer for the query."

    context = "\n".join(retrieved_docs)
    result.append(context)

    # prompt = f"Answer the following query, based only on the given context. Do not add anything from your previous learnings. Do not state in answer that a context is provided to you. If the context seems irrelevant just say 'I don't have an appropriate answer'. query: {query} context: {context}"
    prompt = f"Answer the following query based solely on the provided context. Do not include information from outside the context, and do not mention that a context is provided. If the context does not address the query, respond with 'We're currently in the process of collecting data to provide a comprehensive answer. Thank you for your patience as we work on this. ' If the query includes greetings like 'Good morning' or 'Good evening', respond accordingly. Query: {query} Context: {context}"

    groq_answer = GroqChat(prompt)
    result.append(groq_answer)
    result.append('')
    return result



@app.route('/', methods=['GET', 'POST'])
def index():
    global messages

    if request.method == 'POST':
        # Check if the reset button was clicked
        if request.form.get('reset'):
            messages = []  # Clear the messages
        else:
            # Retrieve the user input from the form
            user_input = request.form.get('user_input')
            if user_input != '':
                messages.append({'type': 'question', 'text': user_input})

                messages = messages[-20:]  # Limit message history to the last 20

                # Database query processing
                conn = get_db_connection()
                try:
                    # Use user_input to retrieve relevant documents
                    retrieved_docs = list(set(get_top_answer(user_input, model, conn)))
                    
                    # Generate a response based on the retrieved documents
                    generated_answer = generate_answer_from_docs(user_input, retrieved_docs)
                    
                    # Save the response to the messages list
                    messages.append({'type': 'answer', 'text': generated_answer[1]})
                except:
                    return render_template('error.html', error_message="error")
                finally:
                    conn.close()  # Ensure the database connection is always closed

    return render_template('index.html', messages=messages)

@app.route('/api/save-feedback', methods=['POST'])
def save_feedback_api():
    feedback_data = request.json  # Parse JSON data from the request

    # Validate the required fields
    if not all(key in feedback_data for key in ('messageId', 'action', 'question', 'answer')):
        return jsonify({'error': 'Missing required fields'}), 400

    # Extract feedback details
    message_id = feedback_data['messageId']
    action = feedback_data['action']
    question = feedback_data['question']
    answer = feedback_data['answer']

    # Save the feedback to the database
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Check if the last inserted row matches the current one (excluding action)
            cur.execute(
                """
                SELECT id FROM feedback
                WHERE message_id = %s AND question = %s AND answer = %s
                ORDER BY id DESC LIMIT 1;
                """,
                (message_id, question, answer)
            )
            existing_row = cur.fetchone()

            # If a matching row exists, delete it
            if existing_row:
                cur.execute(
                    "DELETE FROM feedback WHERE id = %s",
                    (existing_row[0],)
                )

            # Insert the new feedback record
            cur.execute(
                """
                INSERT INTO feedback (message_id, action, question, answer)
                VALUES (%s, %s, %s, %s)
                """,
                (message_id, action, question, answer)
            )
            conn.commit()
        return jsonify({'message': 'Feedback saved successfully'}), 200
    except Exception as e:
        print(f"Error saving feedback: {e}")
        return render_template('error.html', error_message="Failed to save feedback due to a server error."), 500

    finally:
        conn.close()


@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error_message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error_message="Internal server error occurred."), 500

if __name__ == '__main__':
    app.run(debug=True)