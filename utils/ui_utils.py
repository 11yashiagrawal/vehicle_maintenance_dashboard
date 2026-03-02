import os
import streamlit as st
import base64
from utils.config import ASSETS_DIR, STYLE_CSS_FILENAME

def load_global_css():
    
    css_path = os.path.join(ASSETS_DIR, STYLE_CSS_FILENAME)
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def inject_custom_css(css_string: str):
    
    st.markdown(f"<style>{css_string}</style>", unsafe_allow_html=True)

def get_base64_image(image_path: str) -> str:
    
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

def apply_full_page_background():
    load_global_css()
    bg_img_path = os.path.join(ASSETS_DIR, "car-bg.png")
    img_data = get_base64_image(bg_img_path)
    
    if img_data:
        bg_css = f"""
        [data-testid="stAppViewContainer"] {{
            background-image: 
                linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
                url("data:image/png;base64,{img_data}") !important;
        }}
        """
        inject_custom_css(bg_css)

def html_table(headers: list, rows: list[dict]) -> str:
    
    def esc(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    h_style = "color:#00ffff!important;padding:8px 12px;border:1px solid #555;font-weight:600"
    c_style = "color:#ffffff!important;padding:8px 12px;border:1px solid #333"
    
    out = [
        '<div style="overflow-x: auto; width: 100%;">',
        '<table style="width:100%; border-collapse: collapse; border: 1px solid #333; color: #ffffff !important;">',
        "<thead><tr>"
    ]
    for h in headers:
        out.append(f'<th style="{h_style}">{esc(h)}</th>')
    out.append("</tr></thead><tbody>")
    
    for r in rows:
        out.append("<tr>")
        for h in headers:
            out.append(f'<td style="{c_style}">{esc(r.get(h, ""))}</td>')
        out.append("</tr>")
    
    out.append("</tbody></table></div>")
    return "\n".join(out)
