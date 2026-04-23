from PIL import Image
import os

hazy_folder = "data/reside/hazy/"
clean_folder = "data/reside/clean/"
output_folder = "outputs/"
compare_folder = "comparisons/"

os.makedirs(compare_folder, exist_ok=True)

files = os.listdir(output_folder)

for i, name in enumerate(files):

    print(f"Creating {i+1}/{len(files)}")

    clean_name = name.split("_")[0] + ".png"

    hazy = Image.open(os.path.join(hazy_folder, name)).resize((256,256))
    output = Image.open(os.path.join(output_folder, name)).resize((256,256))
    clean = Image.open(os.path.join(clean_folder, clean_name)).resize((256,256))

    canvas = Image.new("RGB", (768,256))

    canvas.paste(hazy, (0,0))
    canvas.paste(output, (256,0))
    canvas.paste(clean, (512,0))

    canvas.save(os.path.join(compare_folder, name))

print("Comparisons saved.")