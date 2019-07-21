import argparse
import os
# Comet will timeout if no internet
try:
    from comet_ml import Experiment
except Exception as e:
    from comet_ml import OfflineExperiment
import torch

def get_params():
    parser = argparse.ArgumentParser(description='Perm')
    # Hparams
    padd = parser.add_argument
    padd('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    padd('--latent_dim', type=int, default=20, metavar='N',
                        help='Latent dim for VAE')
    padd('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    padd('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    padd('--latent_size', type=int, default=50, metavar='N',
                        help='Size of latent distribution (default: 50)')
    padd('--estimator', default='reinforce', const='reinforce',
                    nargs='?', choices=['reinforce', 'lax'],
                    help='Grad estimator for noise (default: %(default)s)')
    padd('--reward', default='soft', const='soft',
                    nargs='?', choices=['soft', 'hard'],
                    help='Reward for grad estimator (default: %(default)s)')

    # Training
    padd('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    padd('--PGD_steps', type=int, default=40, metavar='N',
                        help='max gradient steps (default: 30)')
    padd('--max_iter', type=int, default=20, metavar='N',
                        help='max gradient steps (default: 30)')
    padd('--max_batches', type=int, default=None, metavar='N',
                        help='max number of batches per epoch, used for debugging (default: None)')
    padd('--epsilon', type=float, default=0.5, metavar='M',
			help='Epsilon for Delta (default: 0.1)')
    padd('--LAMBDA', type=float, default=100, metavar='M',
			help='Lambda for L2 lagrange penalty (default: 0.1)')
    padd('--nn_temp', type=float, default=1.0, metavar='M',
                   help='Starting diff. nearest neighbour temp (default: 1.0)')
    padd('--temp_decay_rate', type=float, default=0.9, metavar='M',
                   help='Nearest neighbour temp decay rate (default: 0.9)')
    padd('--temp_decay_schedule', type=float, default=100, metavar='M',
                   help='How many batches before decay (default: 100)')
    padd('--bb_steps', type=int, default=2000, metavar='N',
                        help='Max black box steps per sample(default: 1000)')
    padd('--attack_epochs', type=int, default=10, metavar='N',
                        help='Max numbe of epochs to train G')
    padd('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    padd('--batch_size', type=int, default=256, metavar='S',
                        help='Batch size')
    padd('--embedding_dim', type=int, default=300,
                    help='embedding_dim')
    padd('--embedding_type', type=str, default="non-static",
                    help='embedding_type')
    padd('--test_batch_size', type=int, default=128, metavar='N',
                        help='Test Batch size. 256 requires 12GB GPU memory')
    padd('--test', default=False, action='store_true',
                        help='just test model and print accuracy')
    padd('--deterministic_G', default=False, action='store_true',
                        help='Auto-encoder, no VAE')
    padd('--resample_test', default=False, action='store_true',
                        help='Load model and test resampling capability')
    padd('--resample_iterations', type=int, default=100, metavar='N',
                        help='How many times to resample (default: 100)')
    padd('--clip_grad', default=True, action='store_true',
                        help='Clip grad norm')
    padd('--train_vae', default=False, action='store_true',
                        help='Train VAE')
    padd('--train_ae', default=False, action='store_true',
                        help='Train AE')
    padd('--use_flow', default=False, action='store_true',
                        help='Add A NF to Generator')
    padd('--carlini_loss', default=False, action='store_true',
                        help='Use CW loss function')
    padd('--vanilla_G', default=False, action='store_true',
                        help='Vanilla G White Box')
    padd('--prepared_data',default='dataloader/prepared_data.pickle',
                        help='Test on a single data')

    # Imported Model Params
    padd('--emsize', type=int, default=300,
                        help='size of word embeddings')
    padd('--nhidden', type=int, default=300,
                        help='number of hidden units per layer in LSTM')
    padd('--nlayers', type=int, default=2,
                        help='number of layers')
    padd('--noise_radius', type=float, default=0.2,
                        help='stdev of noise for autoencoder (regularizer)')
    padd('--noise_anneal', type=float, default=0.995,
                        help='anneal noise_radius exponentially by this every 100 iterations')
    padd('--hidden_init', action='store_true',
                        help="initialize decoder hidden state with encoder's")
    padd('--arch_i', type=str, default='300-300',
                        help='inverter architecture (MLP)')
    padd('--arch_g', type=str, default='300-300',
                        help='generator architecture (MLP)')
    padd('--arch_d', type=str, default='300-300',
                        help='critic/discriminator architecture (MLP)')
    padd('--arch_conv_filters', type=str, default='500-700-1000',
                        help='encoder filter sizes for different convolutional layers')
    padd('--arch_conv_strides', type=str, default='1-2-2',
                        help='encoder strides for different convolutional layers')
    padd('--arch_conv_windows', type=str, default='3-3-3',
                        help='encoder window sizes for different convolutional layers')
    padd('--z_size', type=int, default=100,
                        help='dimension of random noise z to feed into generator')
    padd('--temp', type=float, default=1,
                        help='softmax temperature (lower --> more discrete)')
    padd('--enc_grad_norm', type=bool, default=True,
                        help='norm code gradient from critic->encoder')
    padd('--train_emb', type=bool, default=True,
                        help='Train Glove Embeddings')
    padd('--gan_toenc', type=float, default=-0.01,
                        help='weight factor passing gradient from gan to encoder')
    padd('--dropout', type=float, default=0.0,
                        help='dropout applied to layers (0 = no dropout)')
    padd('--useJS', type=bool, default=True,
                        help='use Jenson Shannon distance')
    padd('--perturb_z', type=bool, default=True,
                        help='perturb noise space z instead of hidden c')
    padd('--max_seq_len', type=int, default=200,
                    help='max_seq_len')
    padd('--gamma', type=float, default=0.95,
                    help='Discount Factor')
    padd('--model', type=str, default="lstm_arch",
                    help='classification model name')
    padd('--distance_func', type=str, default="cosine",
                    help='NN distance function')
    padd('--hidden_dim', type=int, default=128,
                    help='hidden_dim')
    padd('--burn_in', type=int, default=500,
                    help='Train VAE burnin')
    padd('--beta', type=float, default=0.,
                    help='Entropy reg')
    padd('--embedding_training', type=bool, default=False,
                    help='embedding_training')
    padd('--seqgan_reward', action='store_true', default=False,
                        help='use seq gan reward')
    padd('--train_classifier', action='store_true', default=False,
                        help='Train Classifier from scratch')
    padd('--diff_nn', action='store_true', default=False,
                        help='Backprop through Nearest Neighbors')
    # Bells
    padd('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    padd('--data_parallel', action='store_true', default=False,
                        help="Use multiple GPUs")
    padd('--save_adv_samples', action='store_true', default=False,
                            help='Write adversarial samples to disk')
    padd('--nearest_neigh_all', action='store_true', default=False,
                          help='Evaluate near. neig. for whole evaluation set')
    padd("--comet", action="store_true", default=False,
            help='Use comet for logging')
    padd("--offline_comet", action="store_true", default=False,
            help='Use comet offline. To upload, after training run: comet-upload file.zip')
    padd("--comet_username", type=str, default="joeybose",
            help='Username for comet logging')
    padd("--comet_apikey", type=str,\
            default="Ht9lkWvTm58fRo9ccgpabq5zV",help='Api for comet logging')
    padd('--debug', default=False, action='store_true',
                        help='Debug')
    padd('--debug_neighbour', default=False, action='store_true',
                        help='Debug nearest neighbour training')
    padd('--load_model', default=False, action='store_true',
                        help='Whether to load a checkpointed model')
    padd('--save_model', default=False, action='store_true',
                        help='Whether to checkpointed model')
    padd('--model_path', type=str, default="saved_models/lstm_torchtext2.pt",\
                        help='where to save/load target model')
    padd('--adv_model_path', type=str, default="saved_models/adv_model.pt",\
                        help='where to save/load adversarial')
    padd('--no_load_embedding', action='store_false', default=True,
                    help='load Glove embeddings')
    padd('--namestr', type=str, default='BMD Text', \
            help='additional info in output filename to describe experiments')
    padd('--dataset', type=str, default="imdb",help='dataset')
    padd('--clip', type=float, default=1, help='gradient clipping, max norm')
    padd('--use_glove', type=str, default="true",
                    help='gpu number')
    args = parser.parse_args()
    args.classes = 2
    args.sample_file = "temp/adv_samples.txt"
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    # Check if settings file
    if os.path.isfile("settings.json"):
        with open('settings.json') as f:
            data = json.load(f)
        args.comet_apikey = data["apikey"]
        args.comet_username = data["username"]

    # Prep file to save adversarial samples
    if args.save_adv_samples:
        now = datetime.datetime.now()
        if os.path.exists(args.sample_file):
            os.remove(args.sample_file)
        with open(args.sample_file, 'w') as f:
            f.write("Adversarial samples starting:\n{}\n".format(now))

    # Comet logging
    args.device = torch.device("cuda" if use_cuda else "cpu")
    if args.comet and not args.offline_comet:
        experiment = Experiment(api_key=args.comet_apikey,
                project_name="black-magic-design",
                workspace=args.comet_username)
    elif args.offline_comet:
        offline_path = "temp/offline_comet"
        if not os.path.exists(offline_path):
            os.makedirs(offline_path)
        from comet_ml import OfflineExperiment
        experiment = OfflineExperiment(
                project_name="black-magic-design",
                workspace=args.comet_username,
                offline_directory=offline_path)

    # To upload offline comet, run: comet-upload file.zip
    if args.comet or args.offline_comet:
        experiment.set_name(args.namestr)
        def log_text(self, msg):
            # Change line breaks for html breaks
            msg = msg.replace('\n','<br>')
            self.log_html("<p>{}</p>".format(msg))
        experiment.log_text = MethodType(log_text, experiment)
        args.experiment = experiment

    return args


