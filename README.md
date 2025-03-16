## Instructions for making predictions:

### Making predictions on Streamlit:

Model deployed for predictions at: https://fashionappuctclassificationgit-pjom4ug42fz4bgneyj93vm.streamlit.app/

### Making predictions using API:

After cloning the repo, run: `python -m uvicorn fashion_api:app --reload`, and visit http://127.0.0.1:8000/docs in your browser.

---

## Instructions for Running

### Running on Kaggle

In order to speed up inference, an intermediate dataset with transforms (resizing, normalization) applied to images was created. It can be found at:  
https://www.kaggle.com/datasets/dyutitmohanty/transformed-imgs-224-224/data  
Simply add the dataset as input on Kaggle.

The images used for example inference in the notebook can be found at:  
https://www.kaggle.com/datasets/dyutitmohanty/pred-imgs-amazon  
Simply add the dataset as input on Kaggle.

---

### Running on Local Device (Outside of Kaggle Notebook)

The following paths in the `FashionProduct_Classification.ipynb` file must be changed:

- `csv_path = "/kaggle/input/fashion-product-images-dataset/fashion-dataset/styles.csv"`  
  Update this path with the location of `styles.csv` on your local device.

- `image_base_dir = '/kaggle/input/fashion-product-images-dataset/fashion-dataset/images'`  
  `images_folder = '/kaggle/input/fashion-product-images-dataset/fashion-dataset/images'`  
  Update these 2 paths with the location of the images folder in the Fashion Product Images Dataset on your local device.

- `transformed_images_folder = '/kaggle/input/transformed-imgs-224-224/kaggle/working/transformed_imgs'`  
  Update this path with the location of the `transformed_imgs` folder after downloading the `transformed_imgs_224_224__` dataset on your device.

- `checkpoint_path = "/kaggle/working/checkpt_model.pth"`  
  `checkpoint_load_path = "/kaggle/working/checkpt_model.pth"`  
  Update these 2 paths with the location of the checkpoint file on your device.

- `prediction_imgs_folder = '/kaggle/input/pred-imgs-amazon/prediction_imgs'`  
  Update this path to location of the folder containing the images you want to perform inference on.

- `intermediate_folder = "/kaggle/working/intermediate"`  
  While performing inference, intermediate images are produced. Set this path to any location that is convenient for storing these intermediate images.
