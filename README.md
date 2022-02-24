# tagflow

Automated myocardial strain estimation using tagged MR images

[![flake8 Actions Status](https://github.com/arnauddhaene/tagflow/actions/workflows/lint.yml/badge.svg)](https://github.com/arnauddhaene/tagflow/actions) 
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/arnauddhaene/tagflow/main/app.py)
![Dependencies](https://img.shields.io/librariesio/github/arnauddhaene/tagflow)
![Docker image](https://img.shields.io/docker/image-size/adhaene/tagflow)


### Dependencies

The dependencies are included in `requirements.txt` and `packages.txt`

### Web-app

Simply run

```bash
streamlit run app.py
```

### Running the code

The folder structure should be as followed:
 
```
├── .github/workflows    <- GitHub Actions linter
├── tagflow              <- Automated tracking web-app src
|   ├── models           <- Neural Network architecture
|   ├── network_saves    <- Checkpointed Neural Networks
|   ├── src              <- Source files
|   ├── widgets          <- Streamlit widgets
|   └── ...              <- Streamlit web-app pages
├── dashboard.py         <- Main file for Streamlit web-app
├── packages.txt         <- Package dependencies for Streamlit Cloud
├── README.md            <- The file you are currently reading
├── requirements.txt     <- Dependencies
└── tox.ini              <- Linting instructions
```
