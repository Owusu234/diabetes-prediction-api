"""
Development server with Flask's built-in debugger
"""
from app import app

if __name__ == '__main__':
    print("Starting DEVELOPMENT server...")
    print("http://localhost:5000")
    print("DO NOT USE IN PRODUCTION\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
