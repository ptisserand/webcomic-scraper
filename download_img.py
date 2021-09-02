#!/usr/bin/env python

import os
import argparse
import requests
import logging
import numpy as np
from bs4 import BeautifulSoup
from requests_html import HTMLSession
import cv2

logger = logging.getLogger("")
def retrieve_img_urls(url: str) -> list:
    """
    Return list of image urls
    """
    ret = []
    session = HTMLSession()
    logger.info("Retrieving web page")
    resp = session.get(url)
    logger.info("Web page retrieved")
    # render javascript
    resp.html.render()    
    logger.info("Web page rendering done")
    soup = BeautifulSoup(resp.html.html, "lxml")
    imgList = soup.find(class_="imgList")
    for img in imgList.find_all("img"):
        ret.append(img.get("data-src"))
    logger.info("Image urls retrieved")
    return ret

def generate_large_img(urls: list, output_filename='large.jpg', save_part=True) -> bytes:
    imgs = []
    count = 0
    full_output_filename = os.path.realpath(output_filename)
    output_dir = os.path.dirname(full_output_filename)
    for url in urls:
        logger.info(f"Downloading {url}")
        resp = requests.get(url)
        raw_data = resp.content
        data = np.frombuffer(raw_data, dtype=np.uint8)
        img = cv2.imdecode(data, flags=cv2.IMREAD_COLOR)
        if save_part is True:
            cv2.imwrite(f'{output_dir}{os.path.sep}{count}.jpg', img)
        imgs.append(img)
        count += 1
    img_v = cv2.vconcat(imgs)
    logger.info("Writing full image")
    cv2.imwrite(f'{full_output_filename}', img_v)
    return img_v




if __name__ == "__main__":
    parser = argparse.ArgumentParser("imgList downloader")
    parser.add_argument("--output", help="Output file to store large image", required=True)
    parser.add_argument("--url", help="URL to scrap", required=True)
    parser.add_argument("--save", help="Keep intermediate image", required=False, action="store_true")
    args = parser.parse_args()

    # log level should come from argument
    log_level = logging.INFO
    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    urls = retrieve_img_urls(args.url)
    generate_large_img(urls=urls, output_filename=args.output, save_part=args.save)

