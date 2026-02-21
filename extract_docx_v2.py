
import zipfile
import xml.etree.ElementTree as ET
import sys

def get_text(filename):
    try:
        with zipfile.ZipFile(filename) as docx:
            tree = ET.XML(docx.read('word/document.xml'))
            namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            text = []
            for paragraph in tree.iterfind('.//w:p', namespace):
                p_text = ''
                for run in paragraph.iterfind('.//w:r', namespace):
                    for text_node in run.iterfind('.//w:t', namespace):
                        if text_node.text:
                            p_text += text_node.text
                if p_text:
                    text.append(p_text)
            return '\n'.join(text)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python extract_docx_v2.py <docx_file>")
        sys.exit(1)
    print(get_text(sys.argv[1]))
