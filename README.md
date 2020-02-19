# CompressionVAE

Data embedding API based on the Variational Autoencoder (VAE), originally proposed by Kingma and Welling https://arxiv.org/abs/1312.6114.

This tool, implemented in TensorFlow 1.x, is designed to work similar to familiar dimensionality reduction methods such as scikit-learn's t-SNE or UMAP, but also go beyond their capabilities in some notable ways, making full use of the VAE as a generative model.

While I decided to call the tool itself CompressionVAE, or CVAE for short, I mainly chose this to give it a unique name.
In practice, it is based on a standard VAE, with the (optional) addition of Inverse Autoregressive Flow (IAF) layers to allow for more flexible posterior distributions.
For details on the IAF layers, I refer you to the original paper: https://arxiv.org/pdf/1606.04934.pdf.

CompressionVAE has **several unique advantages** over the common manifold learning methods like t-SNE and UMAP:
* Rather than just a transformation of the training data, it provides a **reversible and deterministic function**, mapping from data space to embedding space.
* Due to the reversibility of the mapping, the model can be used to **generate new data from arbitrary latent variables**. It also makes them highly suitable as **intermediary representations for downstream tasks**.
* Once a model is trained, it can be reused to transform new data, making it **suitable for use in live settings**.
* Like UMAP, CVAE is **fast and scales much better to large datasets, and high dimensional input and latent spaces**.
* The neural network architecture and training parameters are **highly customisable** through the simple API, allowing more advanced users to tailor the system to their needs.
* VAEs have a **very strong theoretical foundation**, and the learned latent spaces have many desirable properties. There is also extensive literature on different variants, and CVAE can easily be extended to keep up with new research advances.

## Installing CompressionVAE

CompressionVAE is distributed through PyPI under the name `cvae` (https://pypi.org/project/cvae/). To install the latest version, simply run
```
pip install cvae
```
Alternatively, to locally install CompressionVAE, clone this repository and run the following command from the CompressionVAE root directory.
```
pip install -e .
```

## Basic Use Case

To use CVAE to learn an embedding function, we first need to import the cvae library.
```
from cvae import cvae
```

When creating a CompressionVAE object for a new model, it needs to be provided a training dataset. 
For small datasets that fit in memory we can directly follow the sklean convention. Let's look at this case first and take MNIST as an example.

First, load the MNIST data. (Note: this example requires scikit-learn which is not installed with CVAE. You might have to install it first by running `pip install sklearn`.)
```
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, cache=True)
X = mnist.data
```

### Initializing CVAE
Now we can create a CompressionVAE object/model based on this data. The minimal code to do this is
```
embedder = cvae.CompressionVAE(X)
```
By default, this creates a model with two-dimensional latent space, splits the data X randomly into 90% train and 10% validation data, applies feature normalization, and tries to match the model architecture to the input and latent feature dimensions.
It also saves the model in a temporary directory which gets overwritten the next time you create a new CVAE object there.

We will look at customising all this later, but for now let's move on to training.

### Training CVAE
Once a CVAE object is initialised and associated with data, we can train the embedder using its `train` method. This works similar to t-SNE or UMAP's `fit` method.
In the simplest case, we just run 
```
embedder.train()
```
This will train the model, applying automatic learning rate scheduling based on the validation data loss, and stop either when the model converges or after 50k training steps.
We can also stop the training process early through a KeyboardInterrupt (ctrl-c or 'interrupt kernel' in Jupyter notebook). The model will be saved at this point.

It is also possible to stop training and then re-start with different parameters (see more details below).

One note/warning: At the moment, the model can be quite sensitive to initialization (in some rare cases even giving NAN losses). Re-initializing/training the model can improve the results if a training run did not give satisfactory results.

### Embedding data
Once we have a trained model (well, technically even before training, but the results would be random), we can use CVAE to compress data, embedding it into the latent space.
To do this, we use CVAE's `embed` method.

To embed the entire MNIST data:
```
z = embedder.embed(X)
```
But note that other than t-SNE or UMAP, this data does not have to be the same as the training data. It can be new and previously unseen data.

### Visualising the embedding
For two-dimensional latent spaces, CVAE comes with a built-in visualization method, `visualize`. It provides a two-dimensional plot of the embeddings, including class information if available.

To visualize the MNIST embeddings and color them by their respective class, we can run
```
embedder.visualize(z, labels=[int(label) for label in mnist.target])
```
We could also passed the string labels `mnist.target` directly to `labels`, but in that case they would not necessarily be ordered from 0 to 9. 
Optionally, if we pass `labels` as a list of integers like above, we can also pass the `categories` parameter, a list of strings associating names with the labels. In the case of MNIST this is irrelevant since the label and class names are the same.
By default the `visualize` simply displays the plot. By setting the `filename` parameter we can alternatively save the plot to a file.

### Generating data
Finally, we can use CVAE as a generative model, generating data by decoding arbitrary latent vectors using the `decode` method.
If we simply want to 'undo' our MNIST embedding and try to re-create the input data, we can run our embeddings `z` through the `decode` method.
```
X_reconstructed = embedder.decode(z)
```
As a more interesting example, we can use this for data interpolation. Let's say we want to create the data that's halfway between the first and the second MNIST datapoint (a '5' and a '0' respectively).
We can achieve this with the following code
```
import numpy as np
# Combine the two examples and add batch dimension
z_interp = np.expand_dims(0.5*z[0] + 0.5*z[1], axis=0)
# Decode the new latent vector.
X_interp = embedder.decode(z_interp)
```

#### Visualizing the latent space
In the case of image data, such as MNIST, CVAE also has a method that allows us to quickly visualize the latent space as seen through the decoder.
To plot a 20 by 20 grid of reconstructed images, spanning the latent space region [-4, 4] in both x and y, we can run
```
embedder.visualize_latent_grid(xy_range=(-4.0, 4.0),
                               grid_size=20,
                               shape=(28, 28))
```

## Advanced Use Cases
The example above shows the simplest usage of CVAE. However, if desired a user can take much more control over the system and customize the model and training processes.

### Customizing the model
In the previous example we initialised a CompressionVAE with default parameters. If we want more control, we might want to initialise it the following way:
```
embedder = cvae.CompressionVAE(X,
                               train_valid_split=0.99,
                               dim_latent=16,
                               iaf_flow_length=10,
                               cells_encoder=[512, 256, 128],
                               initializer='lecun_normal',
                               batch_size=32,
                               batch_size_test=128,
                               logdir='~/mnist_16d',
                               feature_normalization=False,
                               tb_logging=True)
```
`train_valid_split` controls the random splitting into train and test data. Here 99% of X is used for training, and only 1% is reserved for validation.

Alternatively, to get more control over the data the user can also provide `X_valid` as an input. In this case `train_valid_split` is ignored and the model uses `X` for training and `X_valid` for validation.

`dim_latent` specifies the dimensionality of the latent space.

`iaf_flow_length` controls how many IAF flow layers the model has.

`cells_encoder` determines the number, as well as size of the encoders fully connected layers. In the case above, we have three layers with 512, 256, and 128 units respectively. The decoder uses the mirrored version of this.
If this parameter is not set, CVAE creates a two layer network with sizes adjusted to the input dimension and latent dimension. The logic behind this is very handwavy and arbitrary for now, and I generally recommend setting this manually.

`initializer` controls how the model weights are initialized, with options being `orthogonal` (default), `truncated_normal`, and `lecun_normal`.

`batch_size` and `batch_size_test` determine the batch sizes used for training and testing respectively.

`logdir` specifies the path to the model, and also acts as the model name. The default, `'temp'`, gets overwritten every time it is used, but other model names can be used to save and restore models for later use or even to continue training.

`feature_normalization` tells CVAE whether it should internally apply feature normalization (zero mean, unit variance, based on the training data) or not. If True, the normalisation factors are stored with the model and get applied to any future data.

`tb_logging` determines whether the model writes summaries for TensorBoard during the training process.

### Customizing the training process
In the simple example we called the `train` method without any parameter. A more advanced call might look like
```
embedder.train(learning_rate=1e-4,
               num_steps=2000,
               dropout_keep_prob=0.6,
               test_every=50,
               lr_scheduling=False)
```
`learning_rate` sets the initial learning rate of training.

`num_steps` controls the maximum number of training steps before stopping.

`dropout_keep_prob` determines the keep probability for dropout in the fully connected layers.

`test_every` sets the frequency of test steps.

`lr_scheduling` controls whether learning rate scheduling is applied. If `False`, training continues at `learning_rate` until `num_steps` is reached.

For more arguments/details, for example controlling the details of the learning rate scheduler and the convergence criteria, check the method definition. 

### Using large datasets

Alternatively to providing the input data `X` as a single numpy array, as done with t-SNE and UMAP, CVAE also allows for much larger datasets that do not fit into a single array.

To prepare such a dataset, create a new directory, e.g. `'~/my_dataset'`, and save the training data as individual npy files per example in this directory. 

(Note: the data can also be saved in nested sub-directories, for example one directory per category. CVAE will look through the entire directory tree for npy files.)

When initialising a model based on this kind of data pass the root directory of the dataset as `X`. E.g.
```
embedder = cvae.CompressionVAE(X='~/my_dataset')
```  
Initialising will take slightly longer than if `X` is passed as an array, even for the same number of data points. But this method scales in principle to arbitrarily large datasets, and only loads one batch at a time during training.

### Restarting an existing model

If a CompressionVAE object is initialized with `logdir='temp'` it always starts from a new untrained model, possible overwriting any previous temp model.
However, if a different `logdir` is chosen, the model persists and can be reloaded.

If CompressionVAE is initialized with a `logdir` that already exists and contains parameter and checkpoint files of a previous model, it attempts to restore that model's checkpoints.
In this case, any model specific input parameter (e.g. `dim_latent` and `cells_encoder`) is ignored in favor of the original models parameters.

A restored model can be use straight away to embed or generate data, but it is also possible to continue the training process, picking up from the most recent checkpoint.
