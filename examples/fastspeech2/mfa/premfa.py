from tqdm import tqdm
import os
import argparse
def safemkdir(dirn):
  if not os.path.isdir(dirn):
    os.mkdir(dirn)

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for MFA align")
    parser.add_argument(
        "--dataset-path",
        default="datasets",
        type=str,
        help="Path of LJSpeech or like dataset",
    )
    parser.add_argument(
        "--out-path",
        default="TextGrid",
        type=str,
        help="Directory to create for TextGrid output",
    )
    parser.add_argument(
        "--export-wordlist",
        default="",
        type=str,
        help="If not empty, will export a word list to the file",
    )
    args = parser.parse_args()
    metadpath = args.dataset_path + "/metadata.csv"
    datapath = args.dataset_path + "/wavs"
    dowordlist = len(args.export_wordlist) > 2
    words = list()

    safemkdir(args.out_path)
    print("Preparing dataset for MFA...")
    with open(metadpath,"r") as f:
      for mli in tqdm(f.readlines()):
        lisplit = mli.strip().split("|")

        rawpath = lisplit[0]
        transcr = lisplit[2]


        pfileout = datapath + "/" + rawpath + ".lab"
        fout = open(pfileout,"w")
        fout.write(transcr + "\n")
        fout.close()
        if dowordlist:
          for w in transcr.split(" "):
            if w not in words:
              words.append(w)
    
    if len(words) > 1:
      print("Exporting words...")
      wordsout = open(args.export_wordlist,"w")
      for w in tqdm(words):
        wordsout.write(w + "\n")
      print("Done!")
      wordsout.close()

        

if __name__ == "__main__":
    main()

