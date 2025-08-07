# Chest X-ray Disease Classification

This project is a Streamlit web application that uses a DenseNet121 model to classify diseases from chest X-ray images.

## Files

- `main.py`: The Streamlit app script.
- `densenet121_chestxray_2.pth`: The pretrained PyTorch model weights.
- `requirements.txt`: Python dependencies required to run the app.

## Setup and Running Locally

1. Clone the repository or download the files.
2. Ensure you have Python 3.7 or higher installed.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```
   streamlit run main.py
   ```
5. The app will open in your default web browser. Upload a chest X-ray image to get the predicted disease.

## Deployment

You can deploy this app on platforms that support Streamlit apps, such as:

### Streamlit Cloud

1. Push your project to a GitHub repository.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign in.
3. Create a new app and link it to your GitHub repository.
4. Specify the main file as `main.py`.
5. Deploy the app. The model file `densenet121_chestxray_2.pth` should be included in the repository.

### Heroku

1. Create a `Procfile` with the following content:
   ```
   web: streamlit run main.py --server.port=$PORT --server.enableCORS=false
   ```
2. Push your project to a GitHub repository.
3. Create a new Heroku app and connect it to your GitHub repository.
4. Set buildpacks for Python.
5. Deploy the app.

## Notes

- The model file path is relative, so ensure the model file is in the same directory as `main.py`.
- The app runs on CPU by default.
- For large-scale deployment or production use, consider using GPU-enabled environments and optimizing the model.

## License

This project is provided as-is without warranty.
