import pytesseract

from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

phrase = pytesseract.image_to_string(Image.open('C:\\Users\\pedro.pereira\\Desktop\\Trabalho\\SetupTalk\\Computer Vision\\frase.jpg'), lang='por')

Image('C:\\Users\\pedro.pereira\\Desktop\\Trabalho\\SetupTalk\\Computer Vision\\frase.jpg')

print(phrase)