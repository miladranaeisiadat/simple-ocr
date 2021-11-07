import cv2
import os
import click
import logging
import pytesseract
import re
from pre_processing import *
from matplotlib import pyplot as plt

logging.basicConfig()

class TextRecognition():
    def __init__(self, input_file: str, output_file=None , img_dir = 'images/', verbose= False):
        self.input = input_file
        self.output = output_file
        self.img_dir = img_dir
        self.verbose = verbose
    
    def resize_image(self, image: str):
        '''
        resize the image and grab the new image dimensions
        '''
        # resize the image and grab the new image dimensions
        image = cv2.resize(image, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
        if self.verbose:
            logging.warning('Image has shape : %s ', image.shape)
            b,g,r = cv2.split(image)
            rgb_img = cv2.merge([r,g,b])
            plt.imshow(rgb_img)
            plt.title('ORIGINAL IMAGE')
            plt.show()
        return image

    def ocr(self,image: str):
        custom_config = r'--psm 6 --oem 3'  
        text = pytesseract.image_to_string(image, config=custom_config, lang='eng')
        return str(text)

    def image_to_text(self):
        orginal_image = cv2.imread(self.img_dir + self.input)
        # Resize image
        new_image = self.resize_image(orginal_image)
        # Pre_processing
        grayscale = get_grayscale(new_image)
        image = thresholding(grayscale)
        text = self.ocr(image)
        text = re.sub('[,!@$*§]+', ' ', text)
        text = '\n'.join(text.splitlines())
        return text
        # logging.info(text)
    
    def write_into_text(self, path,  text):
        save_path = path
        try:
            save_file = open(save_path, "w", encoding='utf-8')
            save_file.write(text)
            if save_file:
                logging.warning("file has been saved in to %s", path)
            else:
                logging.warning("couldn't save")

            save_file.close()
        except OSError as e:
            logging.warning("There is not directory.")



@click.command()
@click.option('--input',  help='input file')
@click.option('--output', help='output file directory')
@click.option('--verbose', is_flag=True, help='1 show logs and 0 hide logs')

def run(input, output, verbose):
    # Define type extentions
    extension_file = ['jpeg', 'png', 'jpg', 'pdf']
    type_of_file = str(list(os.path.splitext(input))[1]).split('.')[1]
    # check if verbose is true or not
    if verbose:
        verbose = True
    if str(type_of_file) in extension_file:
        res = TextRecognition(input, verbose= verbose)
        text = res.image_to_text()
        res.write_into_text(str(output), text)
    else:
        logging.warning("file extention not supported")


if __name__ == '__main__':
    run()

