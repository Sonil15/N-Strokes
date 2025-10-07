# app.py

import streamlit as st
from kolam_generator import generate_kolam

st.set_page_config(page_title="Kolam Generator", layout="centered")

# Add comprehensive CSS with unified background colors
st.markdown("""
<style>
    /* Main app background - dark traditional gradient */
    .stApp {
        background: radial-gradient(1200px 800px at 20% 10%, #2b1a16 0%, #1a100d 40%, #120b08 100%);
        background-attachment: fixed;
    }

    /* Match header/navbar to body background */
    .stApp > header,
    header[data-testid="stHeader"],
    [data-testid="stHeader"] > div,
    div[data-testid="stToolbar"],
    .css-18e3th9, .css-1d391kg, .css-1y4p8pa,
    .css-k1vhr4, .css-12oz5g7, .css-1avcm0n, .css-1v0mbdj {
        background: radial-gradient(1200px 800px at 20% 10%, #2b1a16 0%, #1a100d 40%, #120b08 100%) !important;
        background-attachment: fixed !important;
    }

    /* Remove white space at top */
    .css-1y4p8pa { top: 0px; }

    /* Sidebar styling */
    .css-1lcbmhc, .css-1d391kg {
        background: rgba(22, 14, 11, 0.85) !important;
        border-radius: 16px;
        margin: 1rem;
        padding: 1rem;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
        border: 1px solid rgba(255, 179, 0, 0.15);
    }

    /* Main content area */
    .block-container {
        background: rgba(18, 11, 8, 0.9);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 179, 0, 0.18);
    }

    /* Title styling */
    h1 {
        color: #ffb300 !important; /* saffron */
        text-shadow: 0 2px 14px rgba(255, 179, 0, 0.25);
        letter-spacing: 0.5px;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #c93b2b, #ffb300);
        color: #1b0f0b;
        border: 1px solid rgba(255, 179, 0, 0.5);
        border-radius: 28px;
        padding: 0.5rem 2rem;
        font-weight: 700;
        box-shadow: 0 8px 24px rgba(255, 179, 0, 0.25);
        transition: all 0.25s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(255, 179, 0, 0.35);
        filter: brightness(1.05);
    }

    /* Success and error message styling */
    .stSuccess {
        background: linear-gradient(90deg, #1f6b3a, #2d9f63);
        border-radius: 10px;
        color: #f3ffe9;
        border: 1px solid rgba(45, 159, 99, 0.4);
    }

    .stError {
        background: linear-gradient(90deg, #7a1d1d, #b23b3b);
        border-radius: 10px;
        color: #fff6f6;
        border: 1px solid rgba(178, 59, 59, 0.4);
    }

    /* Hide Streamlit menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Header with elegant styling
st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">
             Kolam Generator
        </h1>
        <p style="color: #ffd180; font-size: 1.1rem; font-style: italic; margin-bottom: 2rem;">
            Create algorithmic kolam patterns with a traditional dark aesthetic
        </p>
    </div>
""", unsafe_allow_html=True)

def get_aesthetic_label(ka_value):
    """
    Return the user label based on the aesthetic parameter value
    """
    if ka_value >= 0.9:
        return "Bold & Geometric"
    elif ka_value >= 0.7:
        return "Straight-Line Dominant"
    elif ka_value >= 0.6:
        return "Perfectly Balanced"
    elif ka_value >= 0.5:
        return "Curved & Intricate"
    elif ka_value >= 0.4:
        return "Dense & Complex"
    elif ka_value >= 0.3:
        return "Soft & Flowing"
    else:
        return "Lace-Like Delicate (Advanced)"

# Sidebar with enhanced styling
st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: #FFD700; margin-bottom: 1rem;">Parametric Inputs</h2>
    </div>
""", unsafe_allow_html=True)

kolam_size = st.sidebar.slider(
    " Kolam Size", 
    min_value=3, 
    max_value=101, 
    value=7, 
    step=2
)

# Color and theme controls
kolam_color = st.sidebar.color_picker(" Kolam Color", value="#FFD700")  # Default to golden
theme_choice = st.sidebar.selectbox(" Theme", options=["Light", "Dark"], index=1)  # Default to dark
theme = theme_choice.lower()

# Completion control
complete = st.sidebar.checkbox(" Complete one-stroke kolam", value=False,
    help="If enabled, generate a complete one-stroke kolam. If disabled, you can choose boundary shapes.")

# Boundary shapes (only for non-complete mode)
boundary_type = "vajra"  # Default to 'diamond'
if not complete:
    boundary_type = st.sidebar.selectbox(
        " Boundary Shape",
        options=["vajra", "kona", "matsya", "taranga", "mandala", "prana"],
        index=0,
        help="Applies only in non-complete mode."
    )

# Custom aesthetic parameter slider with Sikku and Kambi labels
st.sidebar.markdown("""
    <div style="
        margin: 1rem 0 0.5rem 0;
        color: #FFD700;
        font-weight: bold;
        font-size: 14px;
    ">
         Aesthetic Parameter
    </div>
""", unsafe_allow_html=True)

# Add custom labels above the slider
st.sidebar.markdown("""
    <div style="
        display: flex;
        justify-content: space-between;
        margin: 0 0 0.5rem 0;
        font-size: 12px;
        color: #FFFFFF;
        font-style: italic;
    ">
        <span><strong>Sikku</strong> (0.00)</span>
        <span><strong>Kambi</strong> (1.00)</span>
    </div>
""", unsafe_allow_html=True)

aesthetic_param = st.sidebar.slider(
    "", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5, 
    step=0.01,
    help="Controls the style of the kolam pattern - from Sikku (curved) to Kambi (straight lines)",
    label_visibility="collapsed"
)

# Display the aesthetic style with enhanced formatting
aesthetic_label = get_aesthetic_label(aesthetic_param)
st.sidebar.markdown(f"""
    <div style="
        background: linear-gradient(45deg, #ff9a8b, #fecfef);
        padding: 0.5rem;
        border-radius: 5px;
        margin: 1rem 0;
        text-align: center;
        color: #8B4513;
        font-weight: bold;
    ">
         Style: {aesthetic_label}
    </div>
""", unsafe_allow_html=True)

# Add explanation for the terms
st.sidebar.markdown("""
    <div style="
        background: rgba(139, 69, 19, 0.1);
        padding: 0.8rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 11px;
        color: #FFFFFF;
        line-height: 1.4;
    ">
        <strong>Traditional Terms:</strong><br>
        • <strong>Sikku</strong>: Curved, flowing kolam patterns<br>
        • <strong>Kambi</strong>: Straight-line, geometric patterns
    </div>
""", unsafe_allow_html=True)

# Remove file format selection and set it to PNG
file_format = "PNG"  # Fixed to PNG format

# Generate button
if st.sidebar.button("Generate Kolam", type="primary"):
    with st.spinner("✨ Creating your kolam..."):
        try:
            # Display a progress bar
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)  # Simulate progress
                import time
                time.sleep(0.01)  # Simulate computation time

            # Generate the kolam and store it in session state
            fig = generate_kolam(
                kolam_size,
                aesthetic_param,
                kolam_color=kolam_color,
                theme=theme,
                complete=complete,
                boundary_type=boundary_type
            )
            st.session_state["kolam_figure"] = fig  # Store the figure in session state
            st.session_state["kolam_metadata"] = {
                "size": kolam_size,
                "aesthetic_label": aesthetic_label,
                "aesthetic_param": aesthetic_param,
                "theme_choice": theme_choice,
                "kolam_color": kolam_color,
                "boundary_type": boundary_type if not complete else None,
                "complete": complete
            }
            st.success("Kolam generated successfully!")

        except Exception as e:
            st.error(f"❌ Error generating kolam: {str(e)}")
            st.markdown("""
                <div style="
                    background: rgba(255, 182, 193, 0.3);
                    padding: 1rem;
                    border-radius: 10px;
                    margin: 1rem 0;
                    text-align: center;
                    color: #8b0000;
                ">
                    🔧 Please check that the kolam_generator module is available and try again.
                </div>
            """, unsafe_allow_html=True)

# Display the Kolam if it exists in session state
if "kolam_figure" in st.session_state:
    metadata = st.session_state["kolam_metadata"]

    # Add CSS animation for the Kolam display
    st.markdown("""
        <style>
            @keyframes fadeIn {
                from {
                    opacity: 0;
                    transform: scale(0.9);
                }
                to {
                    opacity: 1;
                    transform: scale(1);
                }
            }
            .animated-kolam {
                animation: fadeIn 1s ease-in-out;
            }
        </style>
    """, unsafe_allow_html=True)

    # Display the Kolam with animation
    st.markdown(f"""
        <div class="animated-kolam" style="
            background: linear-gradient(135deg, rgba(255,179,0,0.08), rgba(201,59,43,0.08));
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            border-left: 5px solid #ffb300;
        ">
            <h3 style="color: #ffd180; margin-bottom: 0.5rem;">
                 Generated Kolam (Size: {metadata['size']})
            </h3>
            <p style="color: #ffe6ba; font-style: italic;">
                <strong>Aesthetic Style:</strong> {metadata['aesthetic_label']}
                <span style="font-size: 12px;">({metadata['aesthetic_param']:.2f} - {"Sikku" if metadata['aesthetic_param'] < 0.5 else "Kambi"} dominant)</span><br/>
                <strong>Theme:</strong> {metadata['theme_choice']} &nbsp;|&nbsp; <strong>Color:</strong> {metadata['kolam_color']}{"" if metadata['complete'] else f" &nbsp;|&nbsp; <strong>Boundary:</strong> {metadata['boundary_type']}"}
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.pyplot(st.session_state["kolam_figure"])  # Display the stored figure

    # Save the Kolam as a PNG image
    from io import BytesIO
    buffer = BytesIO()
    st.session_state["kolam_figure"].savefig(buffer, format="png", dpi=300)
    buffer.seek(0)

    # Add a download button for the PNG image
    st.download_button(
        label="📥 Download Kolam",
        data=buffer,
        file_name=f"kolam_{metadata['size']}_{metadata['theme_choice'].lower()}_{metadata['aesthetic_param']:.2f}.png",
        mime="image/png"
    )
