import argparse
import app

if __name__ == "__main__":
    deepface_app = app.create_app()
    deepface_app.run(host="0.0.0.0", port=5000)
