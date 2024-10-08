from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from fragpred import predict_fragment_smiles, cleanup_molecule_rdkit, calculate_properties, get_3d_structure
from combine_frag import combine_fragments
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow CORS for all origins

@app.after_request
def after_request(response):
    # Set the Access-Control-Allow-Origin for all responses
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
    return response

@app.route('/get_3d_structure', methods=['POST'])
def get_3d_structure_route():
    data = request.json
    smiles = data.get('smiles')
    if not smiles:
        return jsonify({"error": "SMILES string is required"}), 400
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return jsonify({"error": "Invalid SMILES string"}), 400
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol, nonBondedThresh=500.0)
    pdb_block = Chem.MolToPDBBlock(mol)
    return jsonify({"pdb": pdb_block})

@app.route('/get_2d_structure', methods=['POST'])
def get_2d_structure_route():
    data = request.json
    smiles = data.get('smiles')
    if not smiles:
        return jsonify({"error": "SMILES string is required"}), 400
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return jsonify({"error": "Invalid SMILES string"}), 400
    img = Draw.MolToImage(mol)
    img_path = os.path.join(os.getcwd(), 'molecule.png')
    img.save(img_path)
    return send_file(img_path, mimetype='image/png')

@app.route('/predict_fragment', methods=['POST'])
def predict_fragment():
    data = request.json
    smiles = data.get('smiles')
    protein = data.get('protein')
    if not smiles:
        return jsonify({"error": "SMILES string is required"}), 400
    fragment_smiles = predict_fragment_smiles(smiles, protein)
    cleaned_fragment_smiles = cleanup_molecule_rdkit(fragment_smiles)
    if not cleaned_fragment_smiles:
        return jsonify({"error": "Failed to generate a valid fragment"}), 400
    mol = Chem.MolFromSmiles(cleaned_fragment_smiles)
    if mol is None:
        return jsonify({"error": "Invalid SMILES string"}), 400
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol, nonBondedThresh=500.0)
    fragment_pdb = Chem.MolToPDBBlock(mol)
    properties = calculate_properties(cleaned_fragment_smiles)
    if not fragment_pdb:
        return jsonify({"error": "Failed to generate PDB for fragment"}), 400
    return jsonify({
        "fragment_smiles": cleaned_fragment_smiles,
        "pdb": fragment_pdb,
        "properties": properties
    })

@app.route('/download_pdb', methods=['POST'])
def download_pdb():
    data = request.json
    smiles = data.get('smiles')
    filename = data.get('filename', 'structure.pdb')
    if not smiles:
        return jsonify({"error": "SMILES string is required"}), 400
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return jsonify({"error": "Invalid SMILES string"}), 400
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol, nonBondedThresh=500.0)
    pdb_block = Chem.MolToPDBBlock(mol)
    file_path = os.path.join(os.getcwd(), filename)
    with open(file_path, 'w') as file:
        file.write(pdb_block)
    return send_file(file_path, as_attachment=True, mimetype='chemical/x-pdb', download_name=filename)

@app.route('/combine', methods=['POST'])
def combine():
    frag1_smiles = str(request.json.get('smiles1'))
    frag2_smiles = str(request.json.get('smiles2'))
    try:
        combined_smiles_list = combine_fragments(frag1_smiles, frag2_smiles)
        combined_fragments_with_properties = []
        for smiles in combined_smiles_list:
            properties = calculate_properties(smiles)
            combined_fragments_with_properties.append({
                'smiles': smiles,
                'properties': properties
            })
        return jsonify({'success': True, 'combined_smiles': combined_fragments_with_properties})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
