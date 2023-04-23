#!/usr/bin/env python3

"""
Based on.

https://github.com/ReceiptManager/receipt-parser-legacy/blob/master/receipt_parser_core/enhancer.py
https://tech.trivago.com/post/2015-10-06-python_receipt_parser/
"""

import argparse

import cv2
import numpy as np
from numpy import ndarray, float64
from pytesseract import pytesseract
from scipy.ndimage import rotate

parser = argparse.ArgumentParser(description='Extract text from a receipt')
parser.add_argument('image', type=str, help='Image to extract text from')
args = parser.parse_args()

def rotate_image(img: ndarray, angle=cv2.ROTATE_90_CLOCKWISE) -> ndarray:
    """Can rotate the image by 90 degree intervals.

    Angle must be one of https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga6f45d55c0b1cc9d97f5353a7c8a7aac2"""
    print(f'CV image size: {img.shape}')
    (height, width, _) = img.shape
    if width > height:
        print('rotating')
        img = cv2.rotate(img, angle)

    return img


def correct_skew(image: ndarray, delta: int = 1, limit: int = 5) -> ndarray:
    """
    This comes from https://stackoverflow.com/questions/57964634/python-opencv-skew-correction-for-ocr/57965160#57965160
    which is an implementation of the Projection Profile Method http://www.cvc.uab.es/%7Ebagdanov/pubs/ijdar98.pdf
    """
    def determine_score(arr: ndarray, angle: int) -> (ndarray, float64):
        data = rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles: # TODO can Numba vectorize this scoring? https://coderzcolumn.com/tutorials/python/numba-vectorize-decorator#1
        _, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return corrected


def remove_noise(img: ndarray) -> ndarray:
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    print('Applying gaussianBlur and medianBlur')

    img = cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img = cv2.threshold(cv2.bilateralFilter(img, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img = cv2.adaptiveThreshold(cv2.bilateralFilter(img, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 2)

    return img


def remove_shadows(img: ndarray) -> ndarray:
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)

    return result


def enhance_image(img: ndarray, high_contrast: bool = True, gaussian_blur: bool = True, rotate: bool = True) -> ndarray:
    "Run all the image cleanups to improve OCR results"

    # TODO According to https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html#rescaling it should attempt to be
    # scan image at 300 DPI. Figure out how to calculate DPI for image before applying this scaling factor"""
    img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)

    if rotate:
        rotate_image(img)

    img = correct_skew(img)
    img = remove_shadows(img)

    if high_contrast:
        img = img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if gaussian_blur:
        img = remove_noise(img)

    return img


def run_tesseract(img: ndarray , language: str = "eng") -> str:
    image_data = pytesseract.image_to_string(img, lang=language, timeout=60, config="--psm 6")
    return image_data

def main():

    img = cv2.imread(args.image)
    img = enhance_image(img)
    print(run_tesseract(img, "eng"))


if __name__ == '__main__':
    main()
