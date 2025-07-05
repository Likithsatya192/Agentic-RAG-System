import os
from flask import Flask, render_template, request, jsonify, make_response
from werkzeug.utils import secure_filename
from main import AgenticRAGSystem
from dotenv import load_dotenv

load_dotenv()

UPLOAD_FOLDER = 'documents'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the RAG system
rag_system = AgenticRAGSystem(
    documents_directory=UPLOAD_FOLDER,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/rag', methods=['POST'])
def rag():
    try:
        # Save uploaded files
        files = request.files.getlist('documents')
        uploaded = False
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                uploaded = True
        # Get the query
        query = request.form.get('query', '')
        if not query:
            return make_response(jsonify({'error': 'No query provided.'}), 400)
        # If files were uploaded, reload vector store
        if uploaded:
            rag_system.add_documents(app.config['UPLOAD_FOLDER'])
        # Run the query
        result = rag_system.query(query)
        return make_response(jsonify(result), 200)
    except Exception as e:
        return make_response(jsonify({'error': f'Internal server error: {str(e)}'}), 500)

if __name__ == '__main__':
    app.run(debug=True) 