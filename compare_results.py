import os
from PIL import Image

hazy_dir = "data/reside/hazy"
clean_dir = "data/reside/clean"
pred_dir = "outputs"

save_dir = "comparisons"
os.makedirs(save_dir, exist_ok=True)

files = sorted(os.listdir(pred_dir))[:10]

for name in files:

    # predicted image
    pred = Image.open(os.path.join(pred_dir, name))

    # hazy image
    hazy = Image.open(os.path.join(hazy_dir, name))

    # find clean image
    clean_name = name.split("_")[0] + ".png"
    clean = Image.open(os.path.join(clean_dir, clean_name))

    # resize
    hazy = hazy.resize((256,256))
    clean = clean.resize((256,256))

    # create comparison image
    w, h = hazy.size
    canvas = Image.new("RGB", (w*3, h))

    canvas.paste(hazy, (0,0))
    canvas.paste(pred, (w,0))
    canvas.paste(clean, (w*2,0))

    canvas.save(os.path.join(save_dir, name))

    print("Saved comparison:", name)