from AE.AE_archs import SIMPLE_AE
from maps.DNM import DNM
from utils.utils import load_mnist, save_projection_image


def test_run():
    # Build the DNM model
    deep_map = DNM(input_image_size=(28, 28),
                   latent_size=(100),
                   lattice_size=(20, 30),
                   ae_arch_class=SIMPLE_AE,
                   init_params=None,
                   sigma=None,
                   alpha=None,
                   BATCH_SIZE=64,
                   ae_lr=1e-3,
                   lmbd=1e-6,
                   som_pretrain_lr=0.0005,
                   dnm_map_lr=0.05)

    # Load dataset
    x = load_mnist()

    # Train DNM
    deep_map.train(x_train=x.astype('float32'),
                   dnm_epochs=200,
                   pre_train_epochs=[500, 10000])

    # Save the back-projection image
    mem = deep_map.backproject_map()
    save_projection_image(memory=mem, lattice_size=(20, 30))


if __name__ == "__main__":
    test_run()
