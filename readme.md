# HCIR_Project


### Cloning the repository

Clone the repository

### Install dependencies either in conda environment or python virtual environment
Download spacy language model
```shell
python -m spacy download en_core_web_md
```
Install the requirements using
```shell
pip install -r requirements.txt
```

## Run facerecognition integrated with greeting module 



### Face Recognition
- Make known_faces direcotory
- Place a picture of yourself in the "known_faces" directory with your name.
  For example, osama.jpg
- In main directory
```shell
cd /HCIR_PROJECT-MAIN
```
- Run the module using the command
```shell
python3 pepperassistance.py
```
### Speech Recognition

 - The speech Recognition class recognizes the speech 
 - Features/Methods
  - Text to speech
  - Speech to text
