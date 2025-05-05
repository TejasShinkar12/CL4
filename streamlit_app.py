import streamlit as st
import os
import json

assignment_dir = "assignments"
json_file = "assignments.json"

# Load assignment files and titles
if os.path.exists(json_file):
    with open(json_file, "r") as f:
        assignments = json.load(f)
    assignment_files = [a["file"] for a in assignments]
    assignment_titles = [a["title"] for a in assignments]
else:
    if not os.path.exists(assignment_dir):
        st.error(f"Assignment directory '{assignment_dir}' not found.")
        st.stop()
    assignment_files = [f for f in os.listdir(assignment_dir) if f.endswith(".py")]
    if not assignment_files:
        st.error("No .py files found in the assignment directory.")
        st.stop()
    assignment_titles = [os.path.splitext(f)[0].replace("_", " ").title() for f in assignment_files]

# Streamlit UI
selected_title = st.selectbox("Assignment", assignment_titles)
selected_index = assignment_titles.index(selected_title)
selected_file = assignment_files[selected_index]

file_path = os.path.join(assignment_dir, selected_file)

try:
    with open(file_path, "r") as file:
        code = file.read()
    st.code(code, language="python")
    st.download_button(
        label="Download",
        data=code,
        file_name=selected_file,
        mime="text/plain"
    )
except Exception as e:
    st.error(f"Error reading file: {e}")