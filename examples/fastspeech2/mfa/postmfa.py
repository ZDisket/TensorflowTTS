import textgrid
import yaml
import os
import numpy as np
from tqdm import tqdm
import argparse
import soundfile as sf

def safemkdir(dirn):
  if not os.path.isdir(dirn):
    os.mkdir(dirn)


def secs_to_samples(insecs,insr):
    return round(insecs * insr)

def main():
    parser = argparse.ArgumentParser(description="Read durations from MFA and assign")
    parser.add_argument(
        "--yaml-path",
        default=None,
        type=str,
        help="Path of FastSpeech2 config. Will be used for extracting the hop_size",
    )
    parser.add_argument(
        "--dataset-path",
        default="datasets",
        type=str,
        help="Dataset directory",
    )
    parser.add_argument(
       "--textgrid-path",
        default="TextGrids",
        type=str,
        help="Directory where the TextGrid output from MFA is stored",
    )
    parser.add_argument(
       "--duration-path",
        default="durations",
        type=str,
        help="Directory where the duration output will be stored",
    )
    parser.add_argument(
       "--sample-rate",
        default=22050,
        type=int,
        help="Sample rate of source audio",
    )
    parser.add_argument(
       "--trim",
        default="n",
        type=str,
        help="Whether to apply MFA-based trimming to source audio. This enables mode 2",
    )
    parser.add_argument(
       "--round",
        default="y",
        type=str,
        help="Whether to round durations",
    )
    args = parser.parse_args()
    hopsz = 256
    sarate = args.sample_rate
    
    yapath = args.yaml_path
    inmetadpath = args.dataset_path + "/metadata.csv"
    wavspath = args.dataset_path + "/wavs"
    txgridpath = args.textgrid_path
    doround = True
    if args.round == "n":
      print("Not rounding")
      doround = False
    
    dotrim = False
    if args.trim == "y":
      dotrim = True
      print("Switching to Mode 2")

    tgrids = os.listdir(txgridpath)

    with open(yapath) as file:
        attrs = yaml.load(file)
        hopsz = attrs["hop_size"]

    durationpath = args.duration_path
    safemkdir(durationpath)
    sil_phones = ['sil', 'sp', 'spn', '']
    metafile = open(inmetadpath,"w")
    print("Reading TextGrids...")
    
    for tgp in tqdm(tgrids):
      if not os.path.isfile(txgridpath + "/" + tgp):
        print("Could not find " + tgp)
        if len(wavspath) > 1:
          wavefn = wavspath + "/" + tgp.replace(".TextGrid",".wav")
          if os.path.isfile(wavefn):
            print("Deleting " + wavefn)
            os.remove(wavefn)
        continue

      try:
        tg = textgrid.TextGrid.fromFile(txgridpath + "/" + tgp)
      except:
        print("Failed to read file " + tgp + " , skipping")
        continue
      
      pha = tg[1]
      durations = []
      totdursecs = 0.0
      phs = "{"
      full_wav_path = wavspath + "/" + tgp.replace(".TextGrid",".wav")
      if dotrim:
        sdata, srate = sf.read(full_wav_path)
        
      enc_notsil = False
      
      trim_low_bound = 0.0
      trim_high_bound = 0.0

          
      for interval in pha.intervals:
        mark = interval.mark
        # If we are in Mode 2, we skip start silent phonemes
        if mark in sil_phones:
          mark = "SIL"
          if dotrim and not enc_notsil:
            continue
        else:
          if not enc_notsil:
            trim_low_bound = interval.duration()

          enc_notsil = True
          
        dur = interval.duration()*(sarate/hopsz)
        if doround:
          durations.append(round(dur))
        else:
          durations.append(int(dur))

        
        phs += mark + " "
        totdursecs += interval.duration()
        

      phs += "END"
      durations.append(0)
      phs += "}"
      phs = phs.replace(" }","}")
      trim_high_bound = totdursecs
      
      if dotrim:
        sf.write(full_wav_path,sdata[secs_to_samples(trim_low_bound,srate):secs_to_samples(trim_high_bound,srate)],srate,"PCM_16")
        
    

      
      
      np.save(durationpath + "/" + tgp.replace(".TextGrid","-durations"),np.array(durations))
      metafile.write(tgp.replace(".TextGrid","") + "|" + phs + "|" + phs + "\n")
      


    metafile.close()
  
  

if __name__ == "__main__":
    main()


