
import zipfile
import re
import sys

def extract_all_text(docx_path):
    try:
        with zipfile.ZipFile(docx_path) as z:
            xml_content = z.read('word/document.xml').decode('utf-8')
            # remove XML tags
            text = re.sub('<[^>]+>', ' ', xml_content)
            # collapse whitespace
            text = re.sub('\s+', ' ', text).strip()
            return text
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python aggressive_extract.py <docx_path>")
        sys.exit(1)
        
    print(extract_all_text(sys.argv[1]))
