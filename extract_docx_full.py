
from docx import Document
import sys

def read_docx(file_path):
    try:
        doc = Document(file_path)
        full_text = []

        # Iterate over all block-level elements
        for element in doc.element.body:
             if element.tag.endswith('p'):
                 para = element.text
                 if para:
                     full_text.append(para)
             elif element.tag.endswith('tbl'):
                 # It's a table, iterate rows/cells
                 for row in element.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tr'):
                     row_text = []
                     for cell in row.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tc'):
                         cell_text = ""
                         for p in cell.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
                              # Extract text from p
                              texts = p.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t')
                              for t in texts:
                                  if t.text:
                                      cell_text += t.text
                         if cell_text:
                             row_text.append(cell_text)
                     if row_text:
                         full_text.append(" | ".join(row_text))
                     
        return '\n'.join(full_text)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_docx_full.py <docx_path>")
        sys.exit(1)
        
    print(read_docx(sys.argv[1]))
