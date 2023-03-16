import pandas as pd
import numpy as np
from glob import glob
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
import easyocr
import keras_ocr

plt.style.use('ggplot')

## Prepare data
annot = pd.read_parquet('annot.parquet')
img = pd.read_parquet('img.parquet')
img_fns = glob('train_images/*')

## Identify image ID
img_id = img_fns[0].split('/')[-1].split('.')[0]

# Plot images & image ID
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(plt.imread(img_fns[0]))
ax.axis('off')
plt.show()

fig, axs = plt.subplots(5, 5, figsize=(20, 20))
axs = axs.flatten()
for i in range(25):
    axs[i].imshow(plt.imread(img_fns[i]))
    axs[i].axis('off')
    image_id = img_fns[i].split('/')[-1].rstrip('.jpg')
    n_annot = len(annot.query('image_id == @image_id'))
    axs[i].set_title(f'{image_id} - {n_annot}')
plt.show()

## Method 1: pytesseract
text_pyt = pytesseract.image_to_string(img_fns[5], lang='eng')
print(text_pyt)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(plt.imread(img_fns[11]))
ax.axis('off')
plt.show()

## Method 2: easyocr
reader = easyocr.Reader(['en'], gpu=False)
text_ocr = reader.readtext(img_fns[11])
pd.DataFrame(text_ocr, columns=['bbox','text','confidence'])

## Method 3: keras_ocr
text_keras = keras_ocr.pipeline.Pipeline()
results = pipeline.recognize(img_fns[11])
pd.DataFrame(results[0], columns=['text','bbox'])

fig, ax = plt.subplots(figsize=(10, 10))
keras_ocr.tools.drawAnnotations(plt.imread(img_fns[11]), results[0], ax=ax)
ax.set_title('Keras OCR Result Example')
plt.show()

## Compare easyocr and keras_ocr
reader = easyocr.Reader(['en'], gpu=False)

## easyocr
dfs = []
for img in tqdm(img_fns[:25]):
    result = reader.readtext(img)
    img_id = img.split('.')[0]
    img_df = pd.DataFrame(result, columns=['bbox','text','confidence'])
    img_df['img_id'] = img_id
    dfs.append(img_df)
easyocr_df = pd.concat(dfs)

## keras_ocr
pipeline = keras_ocr.pipeline.Pipeline()

dfs = []
for img in tqdm(img_fns[:25]):
    results = pipeline.recognize([img])
    result = results[0]
    img_id = img.split('/')[-1].split('.')[0]
    img_df = pd.DataFrame(result, columns=['text', 'bbox'])
    img_df['img_id'] = img_id
    dfs.append(img_df)
kerasocr_df = pd.concat(dfs)

## Plot results: easyocr vs. keras_ocr
def plot_compare(img_fn, easyocr_df, kerasocr_df):
    img_id = img_fn.split('/')[-1].split('.')[0]
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))

    easy_results = easyocr_df.query('img_id == @img_id')[['text','bbox']].values.tolist()
    easy_results = [(x[0], np.array(x[1])) for x in easy_results]
    keras_ocr.tools.drawAnnotations(plt.imread(img_fn), 
                                    easy_results, ax=axs[0])
    axs[0].set_title('easyocr results', fontsize=24)

    keras_results = kerasocr_df.query('img_id == @img_id')[['text','bbox']].values.tolist()
    keras_results = [(x[0], np.array(x[1])) for x in keras_results]
    keras_ocr.tools.drawAnnotations(plt.imread(img_fn), 
                                    keras_results, ax=axs[1])
    axs[1].set_title('keras_ocr results', fontsize=24)
    plt.show()

## Loop over images
for img_fn in img_fns[:25]:
    plot_compare(img_fn, easyocr_df, kerasocr_df)