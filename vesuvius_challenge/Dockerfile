# python310 base image
FROM python:3.10.11-buster

# Set the working directory to /app
WORKDIR /app

# Create non-root user
# USER pythonian 

# Start virtual environment
RUN python3 -m venv open3d_env
RUN . open3d_env/bin/activate

# Install pytorch with cuda11.6
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116

# Install other dependencies
RUN pip3 install numpy matplotlib scikit-image scikit-learn ipython jupyterlab opencv-python

# Install open3d
RUN pip3 install open3d

# Libs
RUN apt-get update && apt-get install libgl1-mesa-glx -y

RUN pip3 install tensorboard
# Sanity check
RUN python3 -c "import open3d.ml.torch as ml3d"

# Open jupyter lab on port 9881
# EXPOSE 9881
CMD ["jupyter", "lab", "--port=9881", "--allow-root", "--no-browser"]