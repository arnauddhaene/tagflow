# tagflow

Automated myocardial strain estimation using tagged MR images

[![flake8 Actions Status](https://github.com/arnauddhaene/tagflow/actions/workflows/lint.yml/badge.svg)](https://github.com/arnauddhaene/tagflow/actions) 
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/arnauddhaene/tagflow/main/app.py)
![Dependencies](https://img.shields.io/librariesio/github/arnauddhaene/tagflow)


### Dependencies

The dependencies are included in `requirements.txt` and `packages.txt`

### Web-app

Simply run

```bash
streamlit run app.py
```

### Using Docker to run the web-app

Pull the Docker image using:

```bash
docker pull ghcr.io/arnauddhaene/tagflow:main
```

Subsequently, run the image using:

```bash
docker run ghcr.io/arnauddhaene/tagflow:main
```

You can now view and use the web-app on ```http://localhost:8501/```

### Code structure

The folder structure should be as followed:
 
```
├── .github/workflows    <- GitHub Actions linter
├── sample_data          <- Sample image used for demo and debugging purposes
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
