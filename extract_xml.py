
import zipfile
import shutil
import os

with zipfile.ZipFile("Product Requirements Document.docx", 'r') as zip_ref:
    zip_ref.extract("word/document.xml", "extracted_xml")

shutil.move("extracted_xml/word/document.xml", "document.xml")
shutil.rmtree("extracted_xml")
