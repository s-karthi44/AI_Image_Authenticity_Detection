
import zipfile
import xml.etree.ElementTree as ET
import os
import sys

def extract_text_from_docx(docx_path):
    try:
        with zipfile.ZipFile(docx_path) as z:
            xml_content = z.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            text = []
            for node in tree.iter():
                if node.tag == '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t':
                    if node.text:
                        text.append(node.text)
                elif node.tag == '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p':
                    text.append('\n')
            return ''.join(text)
    except Exception as e:
        return f"Error reading docx: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_docx.py <path_to_docx>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
        
    print(extract_text_from_docx(file_path))
