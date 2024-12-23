
#Overview
This project integrates Pix2Pix and DeepDream models to perform real-time image transformations using your webcam feed. With an interactive GUI, users can adjust various parameters to customize the output, transforming live video into artistic renditions such as comics or Rembrandt-style paintings.

#Features
Real-Time Processing: Apply DeepDream and Pix2Pix transformations to live webcam feeds.
Interactive GUI: Adjust parameters like step size, number of steps, brightness, and contrast using sliders.
Multiple Styles: Switch between different transformation modes, including Face to Comics and Rembrandt styles.
Modular Design: Run each transformation script independently based on your needs.
Prerequisites
Before setting up the project, ensure you have the following installed on your system:

#Operating System: Windows, macOS, or Linux
Python: Version 3.7 or higher
Git: To clone the repository (optional but recommended)
Webcam: For real-time video input
Installation
Follow the steps below to set up the project on your local machine.

1. Clone the Repository
Clone this repository to your local machine using Git:
``
git clone https://github.com/Mohd6288/pix2pix.git
``
Download the module from google drive

pix2pix_face2comic:
https://drive.google.com/file/d/1-QSLnUgIVolOyrtwlUqnFE-XVNwnLn4r/view?usp=sharing 
https://drive.google.com/file/d/1Vo6ofQ3QyV4_9Zabg34mQ8VQCuWs4UZD/view?usp=sharing

pix2pix_rembrandt:
https://drive.google.com/drive/folders/1b7P6-8IDKnjRWWIGCKalF7GwMoaZE4rK?usp=sharing

The models shold be in the same directory thats you clone.

2. create python environment
``
python3 -m venv venv
venv\Scripts\activate
``
3.install requirements

``
pip install -r requirements.txt
``

4. Then you cloud run or other files 

   ``
   python combo_rembrandt_face2comic.py
   `` 




