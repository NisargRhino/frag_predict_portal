import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
import Levenshtein
import pandas as pd
import requests

#from fragpred import predict_fragment_smiles, cleanup_molecule_rdkit, calculate_properties, get_3d_structure

unique_smiles_df = pd.read_csv('unique_smile5.csv')# enter the path of unique_smile5.csv
unique_smiles_list = unique_smiles_df['SMILES'].tolist()

# Function to clean up a molecule
def cleanup_molecule_rdkit(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol)

def calculate_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol_wt = round(Descriptors.MolWt(mol), 2)
    log_p = round(Descriptors.MolLogP(mol), 2)
    h_bond_donors = Descriptors.NumHDonors(mol)
    h_bond_acceptors = Descriptors.NumHAcceptors(mol)
    tpsa = round(Descriptors.TPSA(mol), 2)
    return {
        'molecular_weight': mol_wt,
        'log_p': log_p,
        'hydrogen_bond_donors': h_bond_donors,
        'hydrogen_bond_acceptors': h_bond_acceptors,
        'tpsa': tpsa
    }

def get_3d_structure(smiles):
    #response = requests.post('https://frag-predict-portal-2.onrender.com/get_3d_structure', json={'smiles': smiles})
    response = requests.post('/get_3d_structure', json={'smiles': smiles})
    if response.status_code == 200:
        return response.json().get('pdb')
    else:
        print("Error fetching 3D structure:", response.json())
        return None


def tanimoto_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is None or mol2 is None:
        return 0.0
    
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def string_similarity(smiles1, smiles2):
    distance = Levenshtein.distance(smiles1, smiles2)
    max_len = max(len(smiles1), len(smiles2))
    if max_len == 0:
        return 1.0
    return 1 - (distance / max_len)

def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def find_closest_valid_smiles(predicted_smiles, unique_smiles_list):
    closest_smiles = None
    highest_similarity = -1
    for smiles in unique_smiles_list:
        similarity = string_similarity(predicted_smiles, smiles)
        if similarity > highest_similarity:
            highest_similarity = similarity
            closest_smiles = smiles
    return closest_smiles


def predict_fragment_smiles(smiles, protein, max_length=128):
    model = RobertaForMaskedLM.from_pretrained('protein-models/model-'+str(protein))  # Update with your model path
    tokenizer = RobertaTokenizer.from_pretrained('protein-models/tokenizer-'+str(protein))  # Update with your tokenizer path
    model.eval()

    inputs = tokenizer(smiles, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    logits = outputs.logits
    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_smiles = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    print("intial smiles: ", predicted_smiles)
    if not is_valid_smiles(predicted_smiles):
        print("Predicted SMILES is invalid. Finding the closest valid SMILES...")
        closest_valid_smiles = find_closest_valid_smiles(predicted_smiles, unique_smiles_list)
        predicted_smiles = closest_valid_smiles
        print("new closest predicted smiles: ", predicted_smiles)
    return predicted_smiles


# Example usage
new_drug_smiles = "CC=C(C)C(=O)OC1C(C)=CC23C(=O)C(C=C(COC(C)=O)C(O)C12O)C1C(CC3C)C1(C)C"  # Replace with your input SMILES
predicted_fragment_smiles = predict_fragment_smiles(new_drug_smiles, 'mTOR')
print("Predicted Fragment SMILES:", predicted_fragment_smiles)

actual_fragment_smiles = ""  # Replace with the actual fragment SMILES in order to test accuracy
similarity = tanimoto_similarity(predicted_fragment_smiles, actual_fragment_smiles)
print("Tanimoto Similarity:", similarity)

# Calculate string similarity
string_sim = string_similarity(predicted_fragment_smiles, actual_fragment_smiles)
print("String Similarity:", string_sim)
