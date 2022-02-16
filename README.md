# tagflow

Automated myocardial strain estimation using tagged MR images

### Dependencies

The dependencies are included in `requirements.txt`. You also need to download `torch_track` and `tagsim` repositories (linked below).

### API

To launch the Flask REST API containing routes for tracking inference and ROI segmentation, run:

```bash
python app.py
```

This will expose the following routes locally on `http://127.0.0.1:5000`:

* `track/` which takes the image (time x width x height) and the reference points (Npoints x 2) and outputs the deformation field (Npoints x  2 x time) relative to the reference points. To get them in the image grid, use `points = deformation + reference[:, :, None]`
* `hough/` which computes reference points using the Hough Transform with a circular template. It takes the image as input (time x width x height) as well as parameters relative to `cv2.HoughCircle` and the spacing for the circumferential and radial grid following Figure 3 of Ferdian et al., 2020.

### Web-app

Simply run

```bash
streamlit run dashboard.py
```

### Running the code

The folder structure should be as followed:
 
```
├── .github/workflows    <- GitHub Actions linter
├── tagflow              <- Automated tracking web-app src
|   ├── widgets          <- Streamlit widgets
|   └── ...              <- Streamlit web-app pages
├── torch_track          <- Tracking for tagged MR images 
├── tagsim               <- Pseudo-random MR image generation
├── app.py               <- API routes
├── dashboard.py         <- Main file for Streamlit web-app
├── README.md            <- The file you are currently reading
├── requirements.txt     <- Dependencies
└── tox.ini              <- Linting instructions
```

To find [torch_track](https://github.com/mloecher/tag_tracking/tree/main/torch_track) and [tagsim](https://github.com/mloecher/tag_tracking/tree/main/tagsim).