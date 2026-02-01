"""
Production server using Waitress - Windows compatible
"""
from waitress import serve
from app import app
import socket

def get_local_ip():
    """Get computer's local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

if __name__ == '__main__':
    local_ip = get_local_ip()
    
    print("=" * 60)
    print("DIABETES PREDICTION API - PRODUCTION SERVER")
    print("=" * 60)
    print(f"Server starting on port 5000")
    print(f"Access from this computer: http://localhost:5000")
    print(f"Access from Android emulator: http://10.0.2.2:5000")
    print(f"Access from other devices: http://{local_ip}:5000")
    print(f"Make sure Windows Firewall allows port 5000!")
    print("=" * 60 + "\n")
    
    serve(
        app,
        host='0.0.0.0',  # Listen on all interfaces
        port=5000,
        threads=4,
        channel_timeout=120
    )