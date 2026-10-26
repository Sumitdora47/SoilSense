SoilSense — Soil Analysis and Crop Recommendation

Run locally:
1. Install Python 3.11 or newer.
2. Open a terminal in this folder.
3. Run: pip install -r requirements.txt
4. Run: python app.py
5. Open http://127.0.0.1:8000

Deploy on Render using the included render.yaml:
1. Create a GitHub repository and upload all files and folders from this project.
2. In Render, choose New + > Blueprint and connect that GitHub repository.
3. Render reads render.yaml and creates the free web service.
4. Wait for deployment to finish, then open the public .onrender.com URL.

The app binds to 0.0.0.0 and reads the PORT environment variable for hosted deployment.
The footer credits: Made with ♥ by Sumit.
