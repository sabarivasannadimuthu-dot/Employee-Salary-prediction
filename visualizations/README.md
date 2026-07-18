# Visualizations

Files added:
- Data_visualization.py — static EDA and plots (saves outputs in visualizations/outputs).
- streamlit_app.py — interactive dashboard (run with Streamlit).
- requirements.txt — Python dependencies.

How to run:
1. Install dependencies:
   pip install -r visualizations/requirements.txt

2a. Run static script:
   python visualizations/Data_visualization.py
   -> outputs saved to visualizations/outputs/

2b. Run interactive Streamlit app:
   streamlit run visualizations/streamlit_app.py

Notes:
- The scripts look for the CSV at several common paths; adjust the path in the script if necessary.
