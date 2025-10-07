# health.py
import streamlit as st

st.write("Health check - App is running!")
st.write("Dependencies loaded successfully")

try:
    import numpy as np
    st.write("✅ NumPy loaded")
except:
    st.write("❌ NumPy failed")

try:
    import matplotlib.pyplot as plt
    st.write("✅ Matplotlib loaded")
except:
    st.write("❌ Matplotlib failed")

try:
    from kolam_generator import generate_kolam
    st.write("✅ Kolam generator loaded")
except Exception as e:
    st.write(f"❌ Kolam generator failed: {e}")
