# VideoStitcher

## Description
The aim of this project is to horizontally 'stitch' two videos such that the output video appears to be 'panoramic'. This is achieved using three main techniques:
- Key Point Detection
- Feature Extraction
- Feature Matching

NOTE: I have chosen to use fixed homography for this project. This means the key point detection, feature extraction and feature matching is only done once at the start and the same key points are used for all subsequent video frames. This is to reduce the computation required and should not affect the 'stitching' if the two cameras recording the videos are truly 'fixed'

## Software Versions
- python    `3.6.1`
- cv2       `3.3.0`
- numpy     `1.2.1`
- imtils    `0.4.3`
- tqdm      `4.11.2`
- moviepy   `0.2.3.2`
