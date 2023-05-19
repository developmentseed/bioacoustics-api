from django.shortcuts import render
from django.http import HttpResponse

from rest_framework.permissions import AllowAny
from rest_framework.decorators import api_view, permission_classes
import librosa
from librosa import display as librosa_display
import IPython
from IPython.display import display as ipy_display
import soundfile
import numpy as np
import matplotlib.pyplot as plt

from .frontend import MelSpectrogram, PCENScalingConfig


@api_view(['POST'])
@permission_classes([AllowAny])
def melspectrogram_view(request):
    if request.method == 'POST':
        audio_file = request.FILES.get('audio_file')
        audio = load_audio_window(audio_file, sample_rate=41000)
        mel_spec = plot_audio_melspec(audio, 41000)

        # Set response headers for image
        response = HttpResponse(mel_spec, content_type="image/jpeg")
        response['Content-Disposition'] = 'attachment; filename="image.jpeg"'
        return response
    
    return render(request, 'audio_to_image.html')


def load_audio_window(
    audio_file: any, sample_rate: int, window_size_s: float = 5.0
):
    """Load an audio window."""
    sf = soundfile.SoundFile(audio_file)
    window_size = int(window_size_s * sf.samplerate)
    a = sf.read(window_size)
    
    a = librosa.resample(
        y=a, orig_sr=sf.samplerate, target_sr=sample_rate, res_type='polyphase'
    )
    if len(a.shape) == 2:
        # Downstream ops expect mono audio, so reduce to mono.
        a = a[:, 0]
    return a


def plot_melspec(
    melspec: np.ndarray,
    newfig: bool = False,
    sample_rate: int = 32000,
    frame_rate: int = 100,
    **specshow_kwargs,
):
    """Plot a melspectrogram."""
    if newfig:
        plt.figure(figsize=(12, 5))
    librosa_display.specshow(
        melspec.T,
        sr=sample_rate,
        y_axis='mel',
        x_axis='time',
        hop_length=sample_rate // frame_rate,
        cmap='Greys',
        **specshow_kwargs,
    )

def plot_audio_melspec(
    audio: np.ndarray,
    sample_rate: int,
    newfig: bool = False,
    display_audio=True,
):
    """Plot a melspectrogram from audio."""
    melspec_layer = get_melspec_layer(sample_rate)
    melspec = melspec_layer.apply({}, audio[np.newaxis, :])[0]
    plot_melspec(melspec, newfig=newfig, sample_rate=sample_rate, frame_rate=100)
    plt.show()
    if display_audio:
        ipy_display(IPython.display.Audio(audio, rate=sample_rate))


def get_melspec_layer(sample_rate: int, root=8.0):
    """Creates a melspec layer for easy visualization."""
    # Usage: melspec_layer.apply({}, audio)
    stride = sample_rate // 100
    melspec_layer = MelSpectrogram(  # pytype: disable=wrong-arg-types  # typed-pandas
        96,
        stride,
        2 * stride,
        sample_rate,
        (60.0, sample_rate / 2.0),
        scaling_config=PCENScalingConfig(root=root, bias=0.0),
    )
    return melspec_layer
