from transformers import AutoTokenizer, EsmForProteinFolding


def inference(sequence, out_file_path_name):
  model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
  tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
  model = model.to("cuda")
  output = model.infer(sequence)
  pdb_str = model.output_to_pdb(output)[0]
  
  
  with open(out_file_path_name, "w") as f:
    f.write(pdb_str)
  
  print("PDB saved")



def main():
  sequence = "MTAAADEVRHRDDSIAQDEL"
  inference(sequence, "sample_structure.pdb")


if __name__ == "__main__":
  main()