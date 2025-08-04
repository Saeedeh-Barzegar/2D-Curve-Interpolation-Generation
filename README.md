# An Inception-based Variational Autoencoder for Curves Generation and Interpolation

**Abstract**: In this work, we introduce a novel approach for synthesizing new curves by leveraging Variational Autoencoder (VAE) latent space interpolation. Our method encodes existing ordered point sequences representing curves into a compact latent representation, enabling smooth and meaningful transitions between different curve shapes. By performing controlled interpolations in the learned latent space, we generate diverse, high-quality smooth curves that maintain structural coherence and geometric consistency. The proposed method is particularly useful for applications in shape design, procedural modeling, and data augmentation in geometric learning.

<p align="center">
<img src="https://github.com/Saeedeh-Barzegar/2D-Curve-Interpolation-Generation/blob/main/demonstration.gif?raw=true" alt="An Inception-based Variational Autoencoder for Curves Generation and Interpolation" >
</p>

## How to train the model?

```shell
python main.py --dataset 'path/to/your/training_set'
```

## Citation

You can cite us using this bibliographic reference:

* Barzegar Khalilsaraei, Augsdörfer. **An Inception-based Variational Autoencoder for Curves Generation and Interpolation**. International Conferences in Central Europe on Computer Graphics, Visualization and Computer Vision, 2025.

```bibtex
@article{BarzegarKhalilsaraei2025Inception,
  author={Saeedeh Barzegar Khalilsaraei and Ursula Augsdörfer},
  title={An Inception-based Variational Autoencoder for Curves Generation and Interpolation},
}
```

