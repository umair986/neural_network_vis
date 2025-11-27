# Neural Network Visualizer

An interactive web application that analyzes and visualizes neural network architectures from code. Paste your TensorFlow/Keras or PyTorch code and see a visual representation of your model's architecture with detailed layer information.

## Features

- 🎨 **Interactive Visualization**: Visual graph representation of neural network layers and connections
- 🔍 **Code Analysis**: AI-powered code parsing to extract model architecture
- 📊 **Layer Details**: View detailed information about each layer including parameters, shapes, and descriptions
- 💡 **Real-time Editor**: Built-in Monaco code editor with syntax highlighting
- 🎯 **Framework Support**: Works with TensorFlow/Keras, PyTorch, JAX/Flax, Hugging Face Transformers, FastAI, PyTorch Lightning, MXNet, PaddlePaddle, Scikit-learn, XGBoost, and more
- 📱 **Responsive Design**: Modern UI built with React and shadcn/ui components

## Tech Stack

### Frontend

- React 19
- Monaco Editor (VS Code editor)
- shadcn/ui components
- Tailwind CSS
- @xyflow/react (for graph visualization)
- Axios for API calls

### Backend

- FastAPI (Python web framework)
- MongoDB (database)
- Motor (async MongoDB driver)
- Google Gemini AI (AI-powered code analysis)

## Prerequisites

- Node.js (v16 or higher)
- Python 3.8+
- MongoDB instance
- npm or yarn

## Installation

### Backend Setup

1. Navigate to the backend directory:

```bash
cd backend
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the backend directory with the following variables:

```env
MONGO_URL=your_mongodb_connection_string
DB_NAME=your_database_name
```

5. Start the FastAPI server:

```bash
uvicorn server:app --reload
```

The backend will run on `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:

```bash
cd frontend
```

2. Install dependencies:

```bash
npm install
```

3. Create a `.env` file in the frontend directory:

```env
REACT_APP_BACKEND_URL=http://localhost:8000
```

4. Start the development server:

```bash
npm start
```

The frontend will run on `http://localhost:3001` (or 3000 if available)

## Usage

1. **Open the Application**: Navigate to `http://localhost:3001` in your browser

2. **Enter Your Code**: Paste or type your neural network code in the editor. Example:

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

3. **Analyze**: Click the "Analyze" button to process your code

4. **Explore**:
   - View the visual graph representation of your model
   - Click on nodes to see detailed layer information
   - Hover over connections to understand data flow

## Project Structure

```
NN-visual/
├── backend/
│   ├── server.py           # FastAPI application
│   ├── requirements.txt    # Python dependencies
│   └── .env               # Environment variables
├── frontend/
│   ├── src/
│   │   ├── App.js         # Main application component
│   │   ├── components/    # UI components
│   │   └── lib/           # Utility functions
│   ├── package.json       # Node dependencies
│   └── .env              # Frontend environment variables
└── tests/                 # Test files
```

## API Endpoints

### POST `/api/analyze-code`

Analyzes neural network code and returns structured layer information.

**Request Body:**

```json
{
  "code": "string",
  "framework": "tensorflow|pytorch (optional)"
}
```

**Response:**

```json
{
  "layers": [...],
  "connections": [...],
  "summary": "string"
}
```

## Development

### Available Scripts

**Frontend:**

- `npm start` - Run development server
- `npm build` - Build for production
- `npm test` - Run tests

**Backend:**

- `uvicorn server:app --reload` - Run development server with auto-reload

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Built with React and FastAPI
- UI components from shadcn/ui
- Code editor powered by Monaco Editor
- Graph visualization using @xyflow/react

## Support

For issues, questions, or contributions, please open an issue on GitHub.

---

Made with ❤️ by the Neural Network Visualizer team
