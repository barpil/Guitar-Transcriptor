### Page of dataset on kaggle
https://www.kaggle.com/datasets/mohammedalkooheji/guitar-notes-dataset

### Creating environment
In order to create needed environment please use:
`conda env create -f environment.yml`

### Using API
To test API you can use uvicorn serwer:  
`uvicorn src.api.sound_prediction_api:app --port 8008`

API responds to POST requests at `/predict/` endpoint.  
At this moment API supports only .wav files.  
  
Sample curl command:  
`curl --location 'http://127.0.0.1:8008/predict/' --form 'file=@"[your_.wav_file_path]"'`  
(For tests you can use prepared sample_data from `data/sample_data`)

