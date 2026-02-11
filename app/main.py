"""
Tesla Stock Price Prediction - Flask Application Entry Point.
"""
import sys
from pathlib import Path
from flask import Flask, render_template, send_from_directory

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.backend.api import api_bp

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "frontend" / "templates"),
    static_folder=str(Path(__file__).parent / "frontend" / "static"),
)
app.register_blueprint(api_bp)


@app.route("/")
def index():
    """Serve main dashboard."""
    return render_template("index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    """Serve static files."""
    return send_from_directory(app.static_folder, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
