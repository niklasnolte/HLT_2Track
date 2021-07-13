import aifeynman
from hlt2trk.utils.meta_info import get_data_for_training, locations
from tqdm import tqdm

X_train, y_train, X_val, y_val = get_data_for_training(normalize=True)

basepath = f"{locations.project_root}/feynman_data/"

filename = "MC_preprocessedExp3_{}.txt"

def save(X,Y, suffix=''):
  with open(basepath + filename.format(suffix), 'w') as f:
      for x,y in tqdm(zip(X,Y)):
          lst  = [f'{i}' for i in x] +[f'{y}\n']
          string = ' '.join(lst)
          f.write(string)       
    
save(X_train, y_train, "train")
save(X_val, y_val, "val")

run = aifeynman.run_aifeynman(basepath, filename.format("train"), 0, '14ops.txt')