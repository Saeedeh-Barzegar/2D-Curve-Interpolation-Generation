import argparse
import time
from data.dataset_loader import load_curves
from models.vae import build_vae

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Path to dataset folder')
    args = parser.parse_args()

    start = time.perf_counter()

    input_size = 161
    latent_dim = 32

    X = load_curves(args.dataset, input_size)
    vae, encoder, decoder = build_vae(input_size=input_size, latent_dim=latent_dim)

    vae.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100, restore_best_weights=True)
    vae.fit(X, X, shuffle=True, epochs=2000, batch_size=128, verbose=1, callbacks=[early_stopping])

    vae.save("vae.h5")
    encoder.save("encoder.h5")
    decoder.save("decoder.h5")

    print(f"Training completed in {time.perf_counter() - start:.2f} seconds.")

if __name__ == "__main__":
    main()
