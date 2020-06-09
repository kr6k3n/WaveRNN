from utils.checkpoints import save_checkpoint, restore_checkpoint
import sys
import numpy as np
import time
from pathlib import Path
import os
from utils import data_parallel_workaround
import argparse
from models.tacotron import Tacotron
from utils.paths import Paths
from utils.text.symbols import symbols
from utils.dataset import get_tts_datasets
from utils.neptune_util import *
from utils.display import *
from utils import hparams as hp
import torch.nn.functional as F
from torch import optim
import torch
print(f"imported torch: {torch.__version__}")


def np_now(x: torch.Tensor): return x.detach().cpu().numpy()


def main():
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Train Tacotron TTS')
    parser.add_argument('--force_train', '-f', action='store_true',
                        help='Forces the model to train past total steps')
    parser.add_argument('--force_gta', '-g', action='store_true',
                        help='Force the model to create GTA features')
    parser.add_argument('--use_tpu', '-tpu', action='store_true',
                        help='Use Colab TPU for faster training')
    parser.add_argument('--restore_neptune', '-rn',
                        action='store_true', help='Resume training from neptune')
    parser.add_argument('--force_cpu', '-c', action='store_true',
                        help='Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py',
                        help='The file to use for the hyperparameters')
    args = parser.parse_args()

    hp.configure(args.hp_file)  # Load hparams from file
    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

    if args.restore_neptune:
        print("restoring checkpoints from neptune")
        get_checkpoint_from_neptune()
    neptune = init_experiment()
    print(neptune)
    force_train = args.force_train
    force_gta = args.force_gta

    print("choosing compute device")
    if (not args.force_cpu) and args.use_tpu:
        import torch_xla
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print("If Only You Knew The Power Of The Dark Side...")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        for session in hp.tts_schedule:
            _, _, _, batch_size = session
            if batch_size % torch.cuda.device_count() != 0:
                raise ValueError(
                    '`batch_size` must be evenly divisible by n_gpus!')
    else:
        print("chose cpu :((")
        device = torch.device('cpu')
    print('Using device:', device)

    # Instantiate Tacotron Model
    print('\nInitialising Tacotron Model...\n')
    model = Tacotron(embed_dims=hp.tts_embed_dims,
                     num_chars=len(symbols),
                     encoder_dims=hp.tts_encoder_dims,
                     decoder_dims=hp.tts_decoder_dims,
                     n_mels=hp.num_mels,
                     fft_bins=hp.num_mels,
                     postnet_dims=hp.tts_postnet_dims,
                     encoder_K=hp.tts_encoder_K,
                     lstm_dims=hp.tts_lstm_dims,
                     postnet_K=hp.tts_postnet_K,
                     num_highways=hp.tts_num_highways,
                     dropout=hp.tts_dropout,
                     stop_threshold=hp.tts_stop_threshold).to(device)
    print('Tacotron initialized successfully!')
    optimizer = optim.Adam(model.parameters())
    restore_checkpoint('tts', paths, model, optimizer, create_if_missing=True)

    if not force_gta:
        for i, session in enumerate(hp.tts_schedule):
            current_step = model.get_step()

            r, lr, max_step, batch_size = session

            training_steps = max_step - current_step

            # Do we need to change to the next session?
            if current_step >= max_step:
                # Are there no further sessions than the current one?
                if i == len(hp.tts_schedule)-1:
                    # There are no more sessions. Check if we force training.
                    if force_train:
                        # Don't finish the loop - train forever
                        training_steps = 999_999_999
                    else:
                        # We have completed training. Breaking is same as continue
                        break
                else:
                    # There is a following session, go to it
                    continue

            model.r = r

            simple_table([(f'Steps with r={r}', str(training_steps//1000) + 'k Steps'),
                          ('Batch Size', batch_size),
                          ('Learning Rate', lr),
                          ('Outputs/Step (r)', model.r)])

            train_set, attn_example = get_tts_datasets(
                paths.data, batch_size, r)
            tts_train_loop(paths, model, optimizer, train_set,
                           lr, training_steps, attn_example, neptune)

        print('Training Complete.')
        print('To continue training increase tts_total_steps in hparams.py or use --force_train\n')

    print('Creating Ground Truth Aligned Dataset...\n')

    train_set, attn_example = get_tts_datasets(paths.data, 8, model.r)
    create_gta_features(model, train_set, paths.gta)

    print('\n\nYou can now train WaveRNN on GTA features - use python train_wavernn.py --gta\n')


def tts_train_loop(paths: Paths, model: Tacotron, optimizer, train_set, lr, train_steps, attn_example, neptune):
    # use same device as model parameters
    device = next(model.parameters()).device

    for g in optimizer.param_groups:
        g['lr'] = lr

    total_iters = len(train_set)
    epochs = train_steps // total_iters + 1

    for e in range(1, epochs+1):

        start = time.time()
        running_loss = 0

        # Perform 1 epoch
        for i, (x, m, ids, _) in enumerate(train_set, 1):

            x, m = x.to(device), m.to(device)

            # Parallelize model onto GPUS using workaround due to python bug
            if device.type == 'cuda' and torch.cuda.device_count() > 1:
                m1_hat, m2_hat, attention = data_parallel_workaround(
                    model, x, m)
            else:
                m1_hat, m2_hat, attention = model(x, m)

            m1_loss = F.l1_loss(m1_hat, m)
            m2_loss = F.l1_loss(m2_hat, m)

            loss = m1_loss + m2_loss

            optimizer.zero_grad()
            loss.backward()
            if hp.tts_clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hp.tts_clip_grad_norm)
                if np.isnan(grad_norm):
                    print('grad_norm was NaN!')

            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / i

            speed = i / (time.time() - start)

            step = model.get_step()
            k = step // 1000

            if attn_example in ids:
                idx = ids.index(attn_example)
                attention_img = np_now(attention[idx][:, :160])
                spectrogram_img = np_now(m2_hat[idx])

                save_attention(attention_img, paths.tts_attention/f'{step}')
                save_spectrogram(
                    spectrogram_img, paths.tts_mel_plot/f'{step}', 600)
                neptune.log_image('attention', attention_img)
                neptune.log_image('spectrograms', spectrogram_img)

            neptune.log_metric("step-loss", step, avg_loss)

            if step % hp.tts_checkpoint_every == 0:
                ckpt_name = f'taco_step{k}K'
                save_checkpoint('tts', paths, model, optimizer,
                                name=ckpt_name, is_silent=True)
                save_current_state_to_neptune(neptune)

            msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: {avg_loss:#.4} | {speed:#.2} steps/s | Step: {k}k | '
            stream(msg)

        # Must save latest optimizer state to ensure that resuming training
        # doesn't produce artifacts
        save_checkpoint('tts', paths, model, optimizer, is_silent=True)
        model.log(paths.tts_log, msg)
        print(' ')


def create_gta_features(model: Tacotron, train_set, save_path: Path):
    # use same device as model parameters
    device = next(model.parameters()).device

    iters = len(train_set)

    for i, (x, mels, ids, mel_lens) in enumerate(train_set, 1):

        x, mels = x.to(device), mels.to(device)

        with torch.no_grad():
            _, gta, _ = model(x, mels)

        gta = gta.cpu().numpy()

        for j, item_id in enumerate(ids):
            mel = gta[j][:, :mel_lens[j]]
            mel = (mel + 4) / 8
            np.save(save_path/f'{item_id}.npy', mel, allow_pickle=False)

        bar = progbar(i, iters)
        msg = f'{bar} {i}/{iters} Batches '
        stream(msg)


if __name__ == "__main__":
    main()
