import json

def extract_code(notebook_path, output_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    code_cells = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            code_cells.append(source)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n# %% [cell]\n'.join(code_cells))

if __name__ == '__main__':
    extract_code('prototype.ipynb', 'extracted_logic.py')
